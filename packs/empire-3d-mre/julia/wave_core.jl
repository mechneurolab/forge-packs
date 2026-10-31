# wave_core.jl — numerics for EMPIRE 3D MRE harmonic wave extraction.
#
# Julia port of the pack's former matlab/extractWaves.m and the OSS-SNR code it
# called from magneto (Utilities/MRE/oss_snr: oss_snr_filter.m, convdata.m,
# Strain_SNR.m, OSS_noise.m, smooth3dimage.m). The port follows the MATLAB
# arithmetic step by step, including its quirks, so the outputs match the MATLAB
# pipeline to floating-point tolerance. Quirks kept on purpose are marked
# "MATLAB parity". Two deliberate departures: phase unwrapping uses ROMEO
# instead of extractWaves' Laplacian method, and Y and Z motion are negated
# for Sentinel's UIUC reader.
#
# Pure functions only: no file I/O, no logging. The worker
# (extract_waves_worker.jl) does the NIfTI/.mat I/O around `extract_waves`.
# Dependencies: FFTW (a Sentinel dependency), ROMEO and the
# LinearAlgebra/Statistics stdlibs, all in the bundled Sentinel project.

using FFTW
using ROMEO
using LinearAlgebra
using Statistics

# ─── Otsu threshold (MATLAB multithresh(A), one level) ────────────────────────

"""
    otsu_threshold(A) -> eltype(A)

One-level Otsu threshold, matching MATLAB's `multithresh(A)`: the finite values
are scaled to [0, 1] in the array's precision, binned into 256 bins the way
`imhist` does, and the between-class variance is maximised (ties averaged).
The threshold is mapped back to the data range in the array's precision.
"""
function otsu_threshold(A::AbstractArray{T}) where {T<:AbstractFloat}
    v = filter(!isnan, vec(A))
    fin = filter(isfinite, v)
    isempty(fin) && error("otsu_threshold: no finite values")
    lo, hi = extrema(fin)
    lo == hi && return lo

    nbins = 256
    counts = zeros(Float64, nbins)
    rng = hi - lo
    for x in v
        s = (x - lo) / rng                      # in T, like MATLAB's single math
        # imhist binning for floating data in [0, 1]: round to the nearest of
        # 256 evenly spaced centres; values outside [0, 1] (±Inf) clamp.
        b = isfinite(s) ? floor(Int, Float64(s) * (nbins - 1) + 0.5) : (s > 0 ? nbins - 1 : 0)
        counts[clamp(b, 0, nbins - 1) + 1] += 1
    end
    p = counts ./ sum(counts)
    omega = cumsum(p)
    mu = cumsum(p .* (1:nbins))
    mu_t = mu[end]
    sigma_b2 = (mu_t .* omega .- mu) .^ 2 ./ (omega .* (1 .- omega))

    valid = filter(!isnan, sigma_b2)                # MATLAB max() skips NaN
    maxval = isempty(valid) ? NaN : maximum(valid)
    isfinite(maxval) || return lo
    idx = mean(findall(==(maxval), sigma_b2))
    thresh = idx - 1
    return lo + T(thresh / 255) * rng
end

# ─── Echo ratios and harmonics ────────────────────────────────────────────────

"""
    echo_ratios(img, n_echoes, n_meg, meg_order) -> Array{ComplexF32,5}

`img` is `[nx, ny, nz, n_reps]` with reps ordered echo-fastest, then MEG
direction, then phase offset. Returns echo1 ./ echo2 as
`[nx, ny, nz, 3, n_phase_offsets]` in `[Z, Y, X]` order (`meg_order` maps each
output direction to its MEG acquisition index). Non-finite ratios become 0.
"""
function echo_ratios(img::AbstractArray{ComplexF32,4}, n_echoes::Int, n_meg::Int,
                     meg_order::AbstractVector{<:Integer})
    nx, ny, nz, n_reps = size(img)
    n_reps % (n_echoes * n_meg) == 0 || error(
        "n_reps ($n_reps) is not divisible by n_echoes*n_meg ($n_echoes*$n_meg=$(n_echoes * n_meg))")
    n_echoes >= 2 || error("echo ratio needs at least 2 echoes (n_echoes=$n_echoes)")
    length(meg_order) == 3 || error("meg_order must list 3 MEG indices, got $(meg_order)")
    all(1 .<= meg_order .<= n_meg) || error("meg_order $(meg_order) is out of range for n_meg=$n_meg")
    nph = n_reps ÷ (n_echoes * n_meg)
    raw = reshape(img, nx, ny, nz, n_echoes, n_meg, nph)
    ratio = Array{ComplexF32,5}(undef, nx, ny, nz, 3, nph)
    for p in 1:nph, di in 1:3
        m = meg_order[di]
        @views ratio[:, :, :, di, p] .= raw[:, :, :, 1, m, p] ./ raw[:, :, :, 2, m, p]
    end
    ratio[.!isfinite.(ratio)] .= 0
    return ratio
end

"""
    echo1_magnitudes(img, n_echoes, n_meg, meg_order) -> Array{Float32,5}

Echo-1 magnitude for each output direction and phase offset, in the same
`[nx, ny, nz, 3, n_phase_offsets]` layout as `echo_ratios`. ROMEO weights its
unwrapping path with it.
"""
function echo1_magnitudes(img::AbstractArray{ComplexF32,4}, n_echoes::Int, n_meg::Int,
                          meg_order::AbstractVector{<:Integer})
    nx, ny, nz, n_reps = size(img)
    nph = n_reps ÷ (n_echoes * n_meg)
    raw = reshape(img, nx, ny, nz, n_echoes, n_meg, nph)
    mag = Array{Float32,5}(undef, nx, ny, nz, 3, nph)
    for p in 1:nph, di in 1:3
        @views mag[:, :, :, di, p] .= abs.(raw[:, :, :, 1, meg_order[di], p])
    end
    return mag
end

"""
    unwrap_phases(ratio, mag, mask) -> Array{Float32,5}

Wrapped phase of each echo ratio, masked, unwrapped with ROMEO (Dymerska et
al., MRM 2021; quality-guided region growing weighted by the echo-1 magnitude
`mag`) inside `mask`, masked again. ROMEO's result is congruent with the
wrapped phase, unlike the Laplacian unwrap it replaces, whose error on the
261005 EMPIRE scan was larger than the wave signal; its waves match FSL
prelude's (r >= 0.998 per direction on that scan). `correctglobal` removes the
2π multiple from each volume's mean so offsets don't leak into the harmonic.
"""
function unwrap_phases(ratio::AbstractArray{ComplexF32,5}, mag::AbstractArray{Float32,5},
                       mask::AbstractArray{Bool,3}; on_volume = (i, n) -> nothing)
    nx, ny, nz, nd, nph = size(ratio)
    size(mag) == size(ratio) || error("magnitude is $(size(mag)) but the echo ratios are $(size(ratio))")
    bmask = BitArray(mask)
    fmask = Float32.(mask)
    out = Array{Float32,5}(undef, nx, ny, nz, nd, nph)
    n = nd * nph
    count = 0
    for di in 1:nd, p in 1:nph
        phi = angle.(@view ratio[:, :, :, di, p]) .* fmask
        out[:, :, :, di, p] .= ROMEO.unwrap(phi; mag = mag[:, :, :, di, p], mask = bmask,
                                            correctglobal = true)
        count += 1
        on_volume(count, n)
    end
    out .*= fmask
    return out
end

"""
    harmonic_waves(phase_unwrapped, disp_scale) -> Array{ComplexF32,4}

Scale phase to displacement, reverse the phase-offset axis, FFT along it
(normalised by nph/2) and return the first harmonic, `[nx, ny, nz, 3]`.
"""
function harmonic_waves(phase_unwrapped::AbstractArray{Float32,5}, disp_scale::Real)
    nph = size(phase_unwrapped, 5)
    nph >= 2 || error("need at least 2 phase offsets for a harmonic, got $nph")
    d = reverse(phase_unwrapped; dims = 5) .* Float32(disp_scale)
    f = fft(d, 5) ./ Float32(nph / 2)
    return f[:, :, :, :, 2]
end

# ─── OSS-SNR (magneto oss_snr_filter + Strain_SNR, default settings) ──────────

# Five-point Gauss–Legendre rule on [0, π] (Strain_SNR's averaging over a cycle).
const _GL_WT = let w1 = (322 - 13 * sqrt(70)) / 900, w2 = (322 + 13 * sqrt(70)) / 900
    (w1, w2, 128 / 225, w2, w1)
end
const _GL_XI = let g1 = -(1 / 3 * sqrt(5 + 2 * sqrt(10 / 7))), g2 = -(1 / 3 * sqrt(5 - 2 * sqrt(10 / 7)))
    map(g -> π / 2 + π / 2 * g, (g1, g2, 0.0, -g2, -g1))
end

"""
    gaussian3d(siz, sd) -> Array{Float64,3}

Normalised `siz`³ Gaussian kernel (Strain_SNR's `gaussian3d`).
"""
function gaussian3d(siz::Int, sd::Real)
    r = (siz - 1) ÷ 2
    H = [exp(-(x^2 + y^2 + z^2) / (2 * sd^2)) for x in -r:r, y in -r:r, z in -r:r]
    H[H .< eps(Float64) * maximum(H)] .= 0
    s = sum(H)
    s != 0 && (H ./= s)
    return H
end

"""
    filter3d_mask(stack, mask, H) -> Array{Float64,3}

Masked convolution (Strain_SNR's `filter3Dmask`): each in-mask voxel becomes
the kernel-weighted mean of its in-mask neighbours; others keep their value.
"""
function filter3d_mask(stack::AbstractArray{<:Real,3}, mask::AbstractArray{Bool,3},
                       H::AbstractArray{Float64,3})
    dims = size(stack)
    out = Float64.(stack)
    lf = size(H) .÷ 2
    cn = cld.(size(H), 2)
    all(isodd, size(H)) || error("Filter must be odd-sized")
    Threads.@threads for k in 1:dims[3]
        for j in 1:dims[2], i in 1:dims[1]
            mask[i, j, k] || continue
            lo = (max(1, i - lf[1]), max(1, j - lf[2]), max(1, k - lf[3]))
            hi = (min(dims[1], i + lf[1]), min(dims[2], j + lf[2]), min(dims[3], k + lf[3]))
            la = (max(1, cn[1] - i + 1), max(1, cn[2] - j + 1), max(1, cn[3] - k + 1))
            wsum = 0.0
            acc = 0.0
            for c in lo[3]:hi[3], b in lo[2]:hi[2], a in lo[1]:hi[1]
                mask[a, b, c] || continue
                w = H[la[1] + a - lo[1], la[2] + b - lo[2], la[3] + c - lo[3]]
                wsum += w
                acc += w * stack[a, b, c]
            end
            out[i, j, k] = acc / wsum
        end
    end
    return out
end

"""
    smooth3d(im, sd, mask) -> Array{Float64,3}

magneto `smooth3dimage`: masked Gaussian smoothing with an odd kernel of
`round(5*sd)` voxels (at least 3). MATLAB rounds halves away from zero.
"""
function smooth3d(im::AbstractArray{<:Real,3}, sd::Real, mask::AbstractArray{Bool,3})
    siz = round(Int, 5 * sd, RoundNearestTiesAway)
    iseven(siz) && (siz += 1)
    siz < 3 && (siz = 3)
    return filter3d_mask(im, mask, gaussian3d(siz, sd))
end

# Quadratic 3D polynomial design matrix for an N³ block (weightedfit3dpoly,
# ord = 2). MATLAB parity: the block coordinates come from meshgrid, so the
# "x" column varies along array dimension 2 and "y" along dimension 1.
function _poly2_design(N::Int)
    lf = N ÷ 2
    M = zeros(N^3, 10)
    for (row, I) in enumerate(CartesianIndices((N, N, N)))
        y = I[1] - lf - 1
        x = I[2] - lf - 1
        z = I[3] - lf - 1
        M[row, :] .= (1, x, y, z, x * x, y * y, z * z, x * y, x * z, y * z)
    end
    return M
end

# Solve (M'WM + 1e-10 I) a = M'W d for every column of `D`. MATLAB's `\`
# factors this symmetric positive-definite system by Cholesky.
function _weighted_fit(M::Matrix{Float64}, w::Vector{Float64}, D::Matrix{Float64})
    WM = w .* M
    A = M' * WM
    for t in 1:size(A, 1)
        A[t, t] += 1e-10
    end
    B = WM' * D
    F = cholesky(Symmetric(A); check = false)
    return issuccess(F) ? F \ B : lu(A) \ B
end

# Average octahedral shear strain over one cycle from polynomial coefficients
# (Strain_SNR's octshearstain_curl, OSS part only).
function _oss_from_coeffs(c::Matrix{Float64}, dx::Float64, dy::Float64, dz::Float64)
    # c columns: Ur_u, Ur_v, Ur_w, Ui_u, Ui_v, Ui_w ; rows: polynomial terms.
    g(term, comp, h) = complex(c[term, comp], c[term, comp + 3]) / h
    dudx = g(2, 1, dx); dvdx = g(2, 2, dx); dwdx = g(2, 3, dx)
    dudy = g(3, 1, dy); dvdy = g(3, 2, dy); dwdy = g(3, 3, dy)
    dudz = g(4, 1, dz); dvdz = g(4, 2, dz); dwdz = g(4, 3, dz)
    sxx = dudx; syy = dvdy; szz = dwdz
    sxy = dudy + dvdx; sxz = dudz + dwdx; syz = dvdz + dwdy
    acc = 0.0
    for q in 1:5
        e = cis(_GL_XI[q])
        r(v) = real(v * e)
        octs = 2 / 3 * sqrt(r(sxx - syy)^2 + r(syy - szz)^2 + r(szz - sxx)^2 +
                            3 / 2 * (r(sxy)^2 + r(sxz)^2 + r(syz)^2))
        acc += _GL_WT[q] * octs
    end
    return acc * (π / 2) / π
end

"""
    oss_calc(Ur, Ui, mask, res) -> Array{Float64,3}

Octahedral shear strain map (Strain_SNR's OSS_calc with filtwidth 0.5,
quadratic fit, 3³ blocks widened to 5³ where the mask is sparse).
`Ur`/`Ui` are `[nx, ny, nz, 3]`; `res` is the voxel size in metres.
"""
function oss_calc(Ur::AbstractArray{Float64,4}, Ui::AbstractArray{Float64,4},
                  mask::AbstractArray{Bool,3}, res::NTuple{3,Float64})
    dims = size(mask)
    H = gaussian3d(3, 0.5)
    lf = 1                      # Nfit = 3
    buf = lf + 1
    pdims = dims .+ 2buf
    # Pre-filter, then zero-pad by `buf` on every side.
    U = zeros(pdims..., 6)
    for comp in 1:3
        U[buf+1:buf+dims[1], buf+1:buf+dims[2], buf+1:buf+dims[3], comp] .=
            filter3d_mask(@view(Ur[:, :, :, comp]), mask, H)
        U[buf+1:buf+dims[1], buf+1:buf+dims[2], buf+1:buf+dims[3], comp + 3] .=
            filter3d_mask(@view(Ui[:, :, :, comp]), mask, H)
    end
    pm = falses(pdims)
    pm[buf+1:buf+dims[1], buf+1:buf+dims[2], buf+1:buf+dims[3]] .= mask
    M3 = _poly2_design(3)
    M5 = _poly2_design(5)
    ntrms = 10
    oss = zeros(dims)
    dx, dy, dz = res
    Threads.@threads for ii in 1+buf:dims[1]+buf
        w3 = zeros(27); D3 = zeros(27, 6)
        w5 = zeros(125); D5 = zeros(125, 6)
        for kk in 1+buf:dims[3]+buf, jj in 1+buf:dims[2]+buf
            pm[ii, jj, kk] || continue
            n3 = 0
            for c in -1:1, b in -1:1, a in -1:1
                n3 += pm[ii+a, jj+b, kk+c]
            end
            if n3 <= 2ntrms
                M, w, D, l = M5, w5, D5, 2
            else
                M, w, D, l = M3, w3, D3, 1
            end
            row = 0
            for c in -l:l, b in -l:l, a in -l:l
                row += 1
                w[row] = pm[ii+a, jj+b, kk+c]
                for comp in 1:6
                    D[row, comp] = U[ii+a, jj+b, kk+c, comp]
                end
            end
            coeffs = _weighted_fit(M, w, D)
            oss[ii-buf, jj-buf, kk-buf] = _oss_from_coeffs(coeffs, dx, dy, dz)
        end
    end
    return oss
end

"""
    oss_noise(U, V, W, h, Un, Vn, Wn, mask) -> (ossnoise, noisemask)

Propagation-of-errors OSS noise with central differences (magneto OSS_noise.m).
MATLAB parity: the original builds its "z" neighbours from the dimension-1
neighbours (`upz = U(ii+1,jj,kk)` and so on); that is kept so the SNR matches.
"""
function oss_noise(U, V, W, h::Float64, Un, Vn, Wn, mask::AbstractArray{Bool,3})
    s = size(mask)
    ossnoise = zeros(s)
    noisemask = zeros(Bool, s)          # not a BitArray: threads write neighbouring bits
    Threads.@threads for kk in 2:s[3]-1
        for jj in 2:s[2]-1, ii in 2:s[1]-1
            (mask[ii, jj, kk] && mask[ii+1, jj, kk] && mask[ii-1, jj, kk] &&
             mask[ii, jj+1, kk] && mask[ii, jj-1, kk] &&
             mask[ii, jj, kk+1] && mask[ii, jj, kk-1]) || continue
            noisemask[ii, jj, kk] = true
            upx = U[ii+1, jj, kk]; umx = U[ii-1, jj, kk]
            vpx = V[ii+1, jj, kk]; vmx = V[ii-1, jj, kk]
            wpx = W[ii+1, jj, kk]; wmx = W[ii-1, jj, kk]
            upy = U[ii, jj+1, kk]; umy = U[ii, jj-1, kk]
            vpy = V[ii, jj+1, kk]; vmy = V[ii, jj-1, kk]
            wpy = W[ii, jj+1, kk]; wmy = W[ii, jj-1, kk]
            upz = U[ii+1, jj, kk]; umz = U[ii-1, jj, kk]     # MATLAB parity (see docstring)
            vpz = V[ii+1, jj, kk]; vmz = V[ii-1, jj, kk]
            wpz = W[ii+1, jj, kk]; wmz = W[ii-1, jj, kk]

            so = 2 / 3 * sqrt(((upx - umx) / h / 2 - (vpy - vmy) / h / 2)^2 +
                              ((upx - umx) / h / 2 - (wpz - wmz) / h / 2)^2 +
                              ((vpy - vmy) / h / 2 - (wpz - wmz) / h / 2)^2 +
                              6 * ((upy - umy) / h / 4 + (vpx - vmx) / h / 4)^2 +
                              6 * ((upz - umz) / h / 4 + (wpx - wmx) / h / 4)^2 +
                              6 * ((vpz - vmz) / h / 4 + (wpy - wmy) / h / 4)^2)

            dn = 9 * h^2 * so
            dsdupx = (2upx - 2umx - vpy + vmy - wpz + wmz) / dn
            dsdumx = -dsdupx
            dsdvpy = (-upx + umx + 2vpy - 2vmy - wpz + wmz) / dn
            dsdvmy = -dsdvpy
            dsdwpz = (-upx + umx + 2wpz - 2wmz - vpy + vmy) / dn
            dsdwmz = -dsdwpz

            ds = 6 * h^2 * so
            dsdupy = (upy - umy + vpx - vmx) / ds;  dsdumy = -dsdupy
            dsdupz = (upz - umz + wpx - wmx) / ds;  dsdumz = -dsdupz
            dsdvpx = (upy - umy + vpx - vmx) / ds;  dsdvmx = -dsdvpx
            dsdvpz = (vpz - vmz + wpy - wmy) / ds;  dsdvmz = -dsdvpz
            dsdwpx = (upz - umz + wpx - wmx) / ds;  dsdwmx = -dsdwpx
            dsdwpy = (vpz - vmz + wpy - wmy) / ds;  dsdwmy = -dsdwpy

            v = Un[ii+1, jj, kk]^2 * dsdupx^2 + Un[ii-1, jj, kk]^2 * dsdumx^2 +
                Vn[ii+1, jj, kk]^2 * dsdvpx^2 + Vn[ii-1, jj, kk]^2 * dsdvmx^2 +
                Wn[ii+1, jj, kk]^2 * dsdwpx^2 + Wn[ii-1, jj, kk]^2 * dsdwmx^2 +
                Un[ii, jj+1, kk]^2 * dsdupy^2 + Un[ii, jj-1, kk]^2 * dsdumy^2 +
                Vn[ii, jj+1, kk]^2 * dsdvpy^2 + Vn[ii, jj-1, kk]^2 * dsdvmy^2 +
                Wn[ii, jj+1, kk]^2 * dsdwpy^2 + Wn[ii, jj-1, kk]^2 * dsdwmy^2 +
                Un[ii, jj, kk+1]^2 * dsdupz^2 + Un[ii, jj, kk-1]^2 * dsdumz^2 +
                Vn[ii, jj, kk+1]^2 * dsdvpz^2 + Vn[ii, jj, kk-1]^2 * dsdvmz^2 +
                Wn[ii, jj, kk+1]^2 * dsdwpz^2 + Wn[ii, jj, kk-1]^2 * dsdwmz^2
            ossnoise[ii, jj, kk] = sqrt(v)
        end
    end
    return ossnoise, noisemask
end

"""
    oss_snr(pimg, voxres_mm, mask; on_warn) -> Float64

Octahedral-shear-strain SNR of the unwrapped phase stack `pimg`
(`[nx, ny, nz, 3, nph]`, directions `[Z, Y, X]`), as magneto's
`oss_snr_filter` computes it with Strain_SNR's defaults. Only 4 or 8 phase
offsets are calibrated, as in the original.
"""
function oss_snr(pimg::AbstractArray{<:Real,5}, voxres_mm::NTuple{3,Float64},
                 mask::AbstractArray{Bool,3}; on_warn = msg -> nothing)
    nx, ny, nz, ni, nj = size(pimg)
    nj in (4, 8) || error("OSS-SNR is calibrated for 4 or 8 phase offsets, got $nj")
    idx = findall(mask)

    # Remove the in-mask spatial mean of each volume (MATLAB zeroes the DC bin
    # of an FFT over the masked voxels, which is the same thing).
    phs = zeros(nx, ny, nz, ni, nj)
    for j in 1:nj, i in 1:ni
        vals = Float64[pimg[I, i, j] for I in idx]
        vals .-= mean(vals)
        for (t, I) in enumerate(idx)
            phs[I, i, j] = vals[t]
        end
    end

    wave = (fft(phs, 5) ./ (nj / 2))[:, :, :, :, 2]
    t = [2π * k / nj for k in 0:nj-1]
    err = zeros(nx, ny, nz, ni)
    resid = zeros(nj)
    for i in 1:ni, I in CartesianIndices((nx, ny, nz))
        w = wave[I, i]
        for k in 1:nj
            resid[k] = real(w * cis(t[k])) - phs[I, i, k]
        end
        err[I, i] = std(resid; corrected = true)
    end

    # convdata: reorder [Z, Y, X] -> [X, Y, Z], mask, amplitude/phase in microns.
    fm = Float64.(mask)
    Ur = zeros(nx, ny, nz, 3); Ui = zeros(nx, ny, nz, 3)
    Un = zeros(nx, ny, nz, 3)
    for (d, src) in enumerate((3, 2, 1))
        w = wave[:, :, :, src] .* fm
        A = abs.(w) .* 1e-6
        P = angle.(w)
        Ur[:, :, :, d] .= A .* cos.(P)
        Ui[:, :, :, d] .= A .* sin.(P)
        # ErrorMap in std(misfit) form, smoothed, then 0.5 * 1e-6 (N = 8 factor).
        e = err[:, :, :, src] .* fm
        Un[:, :, :, d] .= 0.5 .* smooth3d(e, 0.5, mask) .* 1e-6
    end

    res = voxres_mm .* 1e-3
    oss = oss_calc(Ur, Ui, mask, res)

    h = mean(res)
    dh = maximum(res) - minimum(res)
    dh / h > 0.05 && on_warn("voxel anisotropy $(round(100 * dh / h; digits = 1))% > 5%; " *
                             "OSS noise uses the mean voxel size")

    oss_nse = zeros(nx, ny, nz)
    noisemask = falses(nx, ny, nz)
    for q in 1:5
        c, s = cos(_GL_XI[q]), sin(_GL_XI[q])
        Ugp = @views Ur[:, :, :, 1] .* c .- Ui[:, :, :, 1] .* s
        Vgp = @views Ur[:, :, :, 2] .* c .- Ui[:, :, :, 2] .* s
        Wgp = @views Ur[:, :, :, 3] .* c .- Ui[:, :, :, 3] .* s
        gp, noisemask = @views oss_noise(Ugp, Vgp, Wgp, h, Un[:, :, :, 1], Un[:, :, :, 2],
                                         Un[:, :, :, 3], mask)
        oss_nse .+= gp .* _GL_WT[q] .* (π / 2) ./ π
    end

    sel = findall(noisemask)
    snr = mean(oss[sel]) / mean(oss_nse[sel])
    return nj == 8 ? snr * 1.35 : snr * 1.35 / sqrt(2.5)
end

# ─── Whole pipeline ───────────────────────────────────────────────────────────

# ─── Brain mask checks ────────────────────────────────────────────────────────

const _FLIP_COMBOS = ((1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3))

"""
    check_mask_orientation(mask, ref) -> Float64

Error if `mask` is flipped relative to `ref` (the magnitude-threshold mask,
which covers the head on the recon grid). A brain mask in the same orientation
overlaps `ref` best as is; if flipping it along some axes overlaps clearly
better (by 0.1), its voxel order doesn't match. Returns the fraction of `mask`
inside `ref`.
"""
function check_mask_orientation(mask::AbstractArray{Bool,3}, ref::AbstractArray{Bool,3})
    frac(m) = count(m .& ref) / count(m)
    c = frac(mask)
    best, best_dims = c, ()
    for dims in _FLIP_COMBOS
        f = frac(reverse(mask; dims = dims))
        f > best && ((best, best_dims) = (f, dims))
    end
    best <= c + 0.1 || error(
        "brain mask does not line up with the reconstruction: " *
        "$(round(100c; digits = 1))% overlap as is, $(round(100best; digits = 1))% " *
        "when flipped along dims $(collect(best_dims))")
    return c
end

"""
    resolve_mask(brain_mask, thresh_mask) -> BitArray{3}

The mask for unwrapping, extraction and the inversion: the approved brain mask
(HD-BET, possibly edited at the gate) when given, after size, non-empty and
orientation checks; otherwise the magnitude threshold.
"""
function resolve_mask(brain_mask, thresh_mask::AbstractArray{Bool,3})
    brain_mask === nothing && return BitArray(thresh_mask)
    size(brain_mask) == size(thresh_mask) || error(
        "brain mask is $(size(brain_mask)) but the reconstruction is $(size(thresh_mask))")
    m = BitArray(brain_mask)
    any(m) || error("brain mask is empty")
    check_mask_orientation(m, thresh_mask)
    return m
end

# ─── Whole pipeline ───────────────────────────────────────────────────────────

"""
    extract_waves(img, voxel_mm; brain_mask=nothing, n_echoes=2, n_meg=3,
                  disp_scale=1.764, meg_order=[3,1,2], on_step, on_warn) -> NamedTuple

Run the wave-extraction pipeline on `img` (`ComplexF32`, `[nx, ny, nz, n_reps]`).
`brain_mask` (`[nx, ny, nz]` Bool) replaces the magnitude-threshold mask.
Returns `(; Xmotion, Ymotion, Zmotion, mask, t2stack, oss_snr, n_phase_offsets)`.
`on_step(step, label)` reports progress on the 7-step scale extractWaves used.
"""
function extract_waves(img::AbstractArray{ComplexF32,4}, voxel_mm::NTuple{3,Float64};
                       brain_mask = nothing,
                       n_echoes::Int = 2, n_meg::Int = 3, disp_scale::Real = 1.764,
                       meg_order::AbstractVector{<:Integer} = [3, 1, 2],
                       on_step = (step, label) -> nothing, on_warn = msg -> nothing)
    on_step(3, "Echo ratios")
    ratio = echo_ratios(img, n_echoes, n_meg, meg_order)

    on_step(4, "Phase unwrapping")
    # Average magnitude across reps; also the anatomical t2stack.
    mag_avg = dropdims(mean(abs.(img); dims = 4); dims = 4)
    mask = resolve_mask(brain_mask, abs.(mag_avg) .> otsu_threshold(abs.(mag_avg)))
    phase_unwrapped = unwrap_phases(ratio, echo1_magnitudes(img, n_echoes, n_meg, meg_order), mask)

    on_step(5, "Harmonic extraction")
    waves = harmonic_waves(phase_unwrapped, disp_scale)

    on_step(6, "OSS-SNR filtering")
    snr = try
        oss_snr(Float64.(phase_unwrapped), voxel_mm, mask; on_warn = on_warn)
    catch err
        err isa ErrorException || rethrow()
        on_warn("OSS-SNR not computed: $(err.msg)")
        NaN
    end

    # Sentinel's read_mre_uiuc expects the axis signs of magneto's legacy
    # proc_3d_empire, which flips the images along dims 1 and 3 before saving.
    # Without those flips Y and Z motion point the wrong way relative to the
    # grid and the inversion drifts upward, so negate them here.
    return (
        Zmotion = -waves[:, :, :, 1],
        Ymotion = -waves[:, :, :, 2],
        Xmotion = waves[:, :, :, 3],
        mask = mask,
        t2stack = mag_avg,
        oss_snr = snr,
        n_phase_offsets = size(phase_unwrapped, 5),
    )
end
