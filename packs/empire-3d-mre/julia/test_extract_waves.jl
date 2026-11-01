# Unit tests for the EMPIRE wave-extraction port.
#
# Run in the bundled Sentinel project (it provides FFTW, ROMEO and JSON3):
#   julia --project=resources/julia/Sentinel resources/packs/empire-3d-mre/julia/test_extract_waves.jl
# The bridge testset also needs resources/julia/ForgeStudio, which it pushes
# onto LOAD_PATH itself.

using Test
using Statistics
using Random
using JSON3

include(joinpath(@__DIR__, "extract_waves_worker.jl"))   # also includes wave_core.jl

@testset "otsu_threshold" begin
    a = Float32[fill(1.0f0, 500); fill(9.0f0, 500)]
    t = otsu_threshold(a)
    @test 1 < t < 9
    @test t isa Float32
    @test otsu_threshold(fill(2.0f0, 10)) == 2.0f0
    # NaN and Inf don't break it.
    b = Float32[a; NaN32; Inf32]
    @test 1 < otsu_threshold(b) < 9
end

@testset "unwrap_phases: ROMEO recovers a wrapped phase up to 2π and stays congruent" begin
    n = 32
    g = [exp(-((i - 16.5)^2 + (j - 16.5)^2 + (k - 16.5)^2) / (2 * 6.0^2)) for i in 1:n, j in 1:n, k in 1:n]
    truth = 5.0 .* g                       # peaks above π, so it wraps
    @test maximum(abs, angle.(cis.(truth)) .- truth) > π
    ratio = reshape(ComplexF32.(cis.(truth)), n, n, n, 1, 1)
    mag = ones(Float32, n, n, n, 1, 1)
    mask = trues(n, n, n)
    un = Float64.(unwrap_phases(ratio, mag, mask)[:, :, :, 1, 1])
    off = un .- truth
    @test maximum(abs, off .- off[1]) < 1e-3            # one global offset...
    @test abs(off[1] / 2π - round(off[1] / 2π)) < 1e-3  # ...a multiple of 2π
    # Outside the mask the result is zero.
    mask[1:4, :, :] .= false
    un2 = unwrap_phases(ratio, mag, mask)
    @test all(iszero, un2[1:4, :, :, 1, 1])
    @test_throws ErrorException unwrap_phases(ratio, ones(Float32, n, n, n, 2, 1), mask)
end

# Cut cost of a 0/1 labelling (1 = sink side) under the original capacities.
function cut_cost(nbr, cap, tr, sink)
    c = 0
    for i in eachindex(tr)
        tr[i] > 0 && sink[i] && (c += tr[i])
        tr[i] < 0 && !sink[i] && (c -= tr[i])
        for d in 1:6
            j = nbr[d, i]
            j != 0 && !sink[i] && sink[j] && (c += cap[d, i])
        end
    end
    return c
end

@testset "max-flow min cut matches brute force on small grids" begin
    rng = MersenneTwister(5)
    # Grids of at most 12 voxels, so every labelling can be enumerated.
    shapes = ((2, 2, 3), (3, 2, 2), (2, 3, 2), (1, 3, 4), (4, 3, 1))
    for trial in 1:200
        mask = rand(rng, shapes[mod1(trial, length(shapes))]...) .< 0.8
        mask[1] = true
        _, nbr = _mask_graph(mask)
        n = size(nbr, 2)
        cap = Int64[nbr[d, i] == 0 ? 0 : rand(rng, 0:9) for d in 1:6, i in 1:n]
        tr = Int64[rand(rng, -12:12) for _ in 1:n]
        side = _bk_mincut!(nbr, copy(cap), copy(tr))
        got = cut_cost(nbr, cap, tr, side .== 2)
        best = minimum(cut_cost(nbr, cap, tr, [((b >> (i - 1)) & 1) == 1 for i in 1:n])
                       for b in 0:(2^n - 1))
        @test got == best
    end
end

@testset "refine_wraps! finds the global minimum and fixes 2π patches" begin
    n = 18
    c = (n - 1) / 2
    grid = [(i, j, k) for i in 0:n-1, j in 0:n-1, k in 0:n-1]
    mask = BitArray([(i - c)^2 + (j - c)^2 + (k - c)^2 < (n / 2 - 1)^2 for (i, j, k) in grid])
    # Smooth phase with every neighbour step below π, so the true unwrap is the
    # unique minimum of the squared neighbour differences.
    truth = [12 * exp(-((i - c)^2 + (j - c + 2)^2 + (k - c)^2) / (2 * 4.5^2)) + 0.3i
             for (i, j, k) in grid] .* mask
    energy(φ) = sum(sum(abs2, diff(φ; dims = a)[mask[ntuple(b -> b == a ? (1:n-1) : (1:n), 3)...] .&
                                                mask[ntuple(b -> b == a ? (2:n) : (1:n), 3)...]])
                    for a in 1:3)
    patch1 = BitArray([(i - 6)^2 + (j - 7)^2 + (k - 8)^2 < 9 for (i, j, k) in grid]) .& mask
    patch2 = BitArray([(i - 11)^2 + (j - 11)^2 + (k - 9)^2 < 6 for (i, j, k) in grid]) .& mask
    patch3 = BitArray([(i - 9)^2 + (j - 12)^2 + (k - 12)^2 < 4 for (i, j, k) in grid]) .& mask
    start = copy(truth)
    start[patch1] .+= 2π
    start[patch2] .-= 2π
    start[patch3] .+= 4π
    @test energy(start) > energy(truth)
    φ = copy(start)
    moved = refine_wraps!(φ, mask)
    @test maximum(abs, φ[mask] .- truth[mask]) < 1e-9
    @test moved == count(patch1 .| patch2 .| patch3)
    @test all(iszero, φ[.!mask])
    @test refine_wraps!(φ, mask) == 0                      # already at the minimum
    @test_throws ErrorException refine_wraps!(φ, trues(4, 4, 4))
    # Float32 volumes, as unwrap_phases passes them, are moved in place too.
    φ32 = Float32.(start)
    refine_wraps!(φ32, mask)
    @test maximum(abs, φ32[mask] .- Float32.(truth[mask])) < 1e-4
end

@testset "unwrap_phases refinement removes a 2π patch in one offset" begin
    n = 24
    c = (n - 1) / 2
    grid = [(i, j, k) for i in 0:n-1, j in 0:n-1, k in 0:n-1]
    mask = BitArray([(i - c)^2 + (j - c)^2 + (k - c)^2 < (n / 2 - 2)^2 for (i, j, k) in grid])
    truth = [4 * cos(2π * (i / 14 + j / 19) - 0.5) + 0.2k for (i, j, k) in grid]
    ratio = reshape(ComplexF32.(cis.(truth)), n, n, n, 1, 1)
    mag = ones(Float32, n, n, n, 1, 1)
    moved = Int[]
    un = unwrap_phases(ratio, mag, mask; on_refine = m -> push!(moved, m))
    off = Float64.(un[:, :, :, 1, 1])[mask] .- truth[mask]
    @test maximum(abs, off .- off[1]) < 1e-3
    @test moved == [0]                                     # ROMEO was already right
    # A patch shifted by 2π (as ROMEO can leave in a low-signal region) is moved back.
    patch = BitArray([(i - 8)^2 + (j - 14)^2 + (k - 12)^2 < 10 for (i, j, k) in grid]) .& mask
    φ = Float32.(un[:, :, :, 1, 1])
    φ[patch] .+= Float32(2π)
    @test refine_wraps!(φ, mask) == count(patch)
    @test maximum(abs, φ .- un[:, :, :, 1, 1]) < 1e-4
    raw = Float64.(unwrap_phases(ratio, mag, mask; refine = false))[:, :, :, 1, 1][mask]
    d = raw .- Float64.(un[:, :, :, 1, 1])[mask]
    @test maximum(abs, d .- d[1]) < 1e-4                  # same up to the global 2π choice
end

@testset "echo1_magnitudes follows meg_order" begin
    nx, ny, nz, ne, nm, nph = 2, 2, 2, 2, 3, 2
    img = ones(ComplexF32, nx, ny, nz, ne * nm * nph)
    raw = reshape(img, nx, ny, nz, ne, nm, nph)
    for m in 1:nm, p in 1:nph
        raw[:, :, :, 1, m, p] .= ComplexF32(0, 10m + p)   # magnitude 10m + p
    end
    mag = echo1_magnitudes(img, ne, nm, [3, 1, 2])
    @test size(mag) == (nx, ny, nz, 3, nph)
    @test mag[1, 1, 1, 1, 2] == 32   # Z ← MEG 3
    @test mag[1, 1, 1, 2, 2] == 12   # Y ← MEG 1
    @test mag[1, 1, 1, 3, 1] == 21   # X ← MEG 2
end

@testset "brain mask checks" begin
    ref = falses(10, 10, 10); ref[2:9, 2:9, 2:9] .= true
    good = falses(10, 10, 10); good[3:8, 3:8, 3:8] .= true
    @test resolve_mask(nothing, ref) == ref
    @test resolve_mask(good, ref) == good
    # A mask that only fits the head when flipped along dim 1.
    off = falses(10, 10, 10); off[1:4, 3:8, 3:8] .= true; off[1, :, :] .= true
    ref2 = falses(10, 10, 10); ref2[6:10, 2:9, 2:9] .= true
    @test_throws ErrorException resolve_mask(off, ref2)
    @test_throws ErrorException resolve_mask(falses(10, 10, 10), ref)
    @test_throws ErrorException resolve_mask(trues(10, 10, 9), ref)
end

@testset "echo_ratios orders directions by meg_order and zeroes non-finite" begin
    nx, ny, nz, ne, nm, nph = 2, 2, 2, 2, 3, 4
    img = ones(ComplexF32, nx, ny, nz, ne * nm * nph)
    raw = reshape(img, nx, ny, nz, ne, nm, nph)
    for m in 1:nm, p in 1:nph
        raw[:, :, :, 1, m, p] .= ComplexF32(10m + p)
    end
    raw[1, 1, 1, :, 2, 1] .= 0                  # 0/0 → NaN → 0
    r = echo_ratios(img, ne, nm, [3, 1, 2])
    @test size(r) == (nx, ny, nz, 3, nph)
    @test r[2, 2, 2, 1, 3] == ComplexF32(33)   # Z ← MEG 3
    @test r[2, 2, 2, 2, 3] == ComplexF32(13)   # Y ← MEG 1
    @test r[2, 2, 2, 3, 3] == ComplexF32(23)   # X ← MEG 2
    @test r[1, 1, 1, 3, 1] == 0
    @test_throws ErrorException echo_ratios(img[:, :, :, 1:5], ne, nm, [3, 1, 2])
end

@testset "harmonic_waves extracts the first harmonic" begin
    nph = 4
    A, θ, scale = 0.7, 0.4, 1.764
    pu = zeros(Float32, 1, 1, 1, 3, nph)
    for p in 1:nph
        pu[1, 1, 1, :, p] .= A * cos(2π * (p - 1) / nph + θ)
    end
    w = harmonic_waves(pu, scale)
    @test size(w) == (1, 1, 1, 3)
    @test abs(w[1]) ≈ A * scale rtol = 1e-5
    # Reversing the offsets conjugates the phase relative to a plain FFT.
    @test angle(w[1]) ≈ 2π / nph - θ atol = 1e-5
end

@testset "gaussian3d, filter3d_mask and smooth3d" begin
    H = gaussian3d(3, 0.5)
    @test size(H) == (3, 3, 3)
    @test sum(H) ≈ 1
    @test H[2, 2, 2] == maximum(H)
    mask = falses(6, 6, 6); mask[2:5, 2:5, 2:5] .= true
    c = fill(3.0, 6, 6, 6)
    @test filter3d_mask(c, mask, H) ≈ c        # constants are preserved, even at the mask edge
    x = randn(6, 6, 6)
    f = filter3d_mask(x, mask, H)
    @test f[.!mask] == x[.!mask]               # outside the mask is untouched
    @test smooth3d(x, 0.5, mask) == f          # round(2.5) → 3 (ties away from zero)
end

@testset "quadratic fit follows meshgrid axes" begin
    M = _poly2_design(3)
    @test size(M) == (27, 10)
    # Data that grows along array dimension 2 is the "x" term (column 2).
    d = [Float64(j) for i in 1:3, j in 1:3, k in 1:3]
    c = _weighted_fit(M, ones(27), reshape(vec(d), :, 1))
    @test c[2] ≈ 1 atol = 1e-8
    @test abs(c[3]) < 1e-8 && abs(c[4]) < 1e-8
end

@testset "oss_calc: uniform translation has no shear strain" begin
    dims = (8, 8, 8)
    mask = trues(dims)
    Ur = zeros(dims..., 3); Ui = zeros(dims..., 3)
    Ur[:, :, :, 1] .= 2.0; Ui[:, :, :, 3] .= -1.0
    oss = oss_calc(Ur, Ui, mask, (1e-3, 1e-3, 1e-3))
    @test maximum(abs, oss) < 1e-6
    # A shear gradient gives a non-zero, finite strain.
    for j in 1:8
        Ur[:, j, :, 1] .= 1e-6 * j
    end
    oss = oss_calc(Ur, Ui, mask, (1e-3, 1e-3, 1e-3))
    @test all(isfinite, oss) && maximum(oss) > 0
end

# A synthetic EMPIRE acquisition: echo 2 is the magnitude, echo 1 carries a
# displacement phase A_d * g(r) * cos(2π p / nph + θ_d) for each MEG direction,
# plus a little phase noise.
function synthetic_img(; n = 32, nph = 4, amps = (0.9, 0.6, 0.3), θs = (0.2, 1.0, -0.5))
    rng = MersenneTwister(7)
    c = (n + 1) / 2
    r2 = [((i - c)^2 + (j - c)^2 + (k - c)^2) for i in 1:n, j in 1:n, k in 1:n]
    mag = Float32.(ifelse.(r2 .< (0.4n)^2, 100.0, 1.0))
    g = exp.(-r2 ./ (2 * (n / 8)^2))
    ne, nm = 2, 3
    img = Array{ComplexF32,4}(undef, n, n, n, ne * nm * nph)
    raw = reshape(img, n, n, n, ne, nm, nph)
    # meg_order [3,1,2]: output Z,Y,X come from MEG 3,1,2.
    dir_of_meg = Dict(3 => 1, 1 => 2, 2 => 3)
    for p in 1:nph, m in 1:nm
        d = dir_of_meg[m]
        φ = amps[d] .* g .* cos(2π * (p - 1) / nph + θs[d])
        raw[:, :, :, 1, m, p] .= mag .* cis.(Float32.(φ .+ 0.01 .* randn(rng, n, n, n)))
        raw[:, :, :, 2, m, p] .= mag
    end
    return img, amps, n
end

@testset "extract_waves end to end" begin
    img, amps, n = synthetic_img()
    steps = Int[]
    warns = String[]
    res = extract_waves(img, (2.0, 2.0, 2.0); on_step = (s, _) -> push!(steps, s),
                        on_warn = m -> push!(warns, m))
    @test steps == [3, 4, 5, 6]
    @test size(res.Xmotion) == (n, n, n)
    @test eltype(res.Xmotion) == ComplexF32
    @test res.n_phase_offsets == 4
    @test res.n_wraps_refined == 0                         # nothing wraps in this phantom
    c = n ÷ 2 + 1
    @test res.mask[c, c, c] && !res.mask[1, 1, 1]
    # Centre amplitude ≈ A_d * scale (g ≈ 1 there).
    @test abs(res.Zmotion[c, c, c]) ≈ amps[1] * 1.764 rtol = 0.1
    @test abs(res.Ymotion[c, c, c]) ≈ amps[2] * 1.764 rtol = 0.1
    @test abs(res.Xmotion[c, c, c]) ≈ amps[3] * 1.764 rtol = 0.1
    # Y and Z are negated for Sentinel's UIUC reader; X is not.
    w = harmonic_waves(unwrap_phases(echo_ratios(img, 2, 3, [3, 1, 2]),
                                     echo1_magnitudes(img, 2, 3, [3, 1, 2]), res.mask), 1.764)
    @test res.Zmotion == -w[:, :, :, 1]
    @test res.Ymotion == -w[:, :, :, 2]
    @test res.Xmotion == w[:, :, :, 3]
    @test res.t2stack[c, c, c] ≈ 100
    @test isfinite(res.oss_snr) && res.oss_snr > 0
    @test isempty(warns)
end

@testset "extract_waves uses the brain mask" begin
    img, _, n = synthetic_img()
    c = n ÷ 2 + 1
    bm = falses(n, n, n); bm[c-6:c+6, c-6:c+6, c-6:c+6] .= true
    res = extract_waves(img, (2.0, 2.0, 2.0); brain_mask = bm)
    @test res.mask == bm
    @test all(iszero, res.Xmotion[.!bm])
    @test_throws ErrorException extract_waves(img, (2.0, 2.0, 2.0); brain_mask = falses(n, n, n))
end

@testset "OSS-SNR needs 4 or 8 phase offsets" begin
    img, _, _ = synthetic_img(nph = 6)
    warns = String[]
    res = extract_waves(img, (2.0, 2.0, 2.0); on_warn = m -> push!(warns, m))
    @test isnan(res.oss_snr)
    @test any(occursin("OSS-SNR", w) for w in warns)
end

@testset "worker helpers" begin
    @test subject_name("/data/run/stage_2_recon") == "stage_2_recon"
    @test subject_name("/data/run/recon.v2/") == "recon"
    names = ["img_Slice0_Rep3_Avg0_Echo0_Phase0_mag.nii", "img_Slice0_Rep1_Avg0_Echo0_Phase0_mag.nii",
             "img_Slice0_Rep1_Avg0_Echo0_Phase0_phs.nii", "notes.txt"]
    @test scan_recon_files(names) == ([0], [1, 3])
    bm, recon = split_inputs(["/r/x_brain_mask.nii", "/r/img_Slice0_Rep1_Avg0_Echo0_Phase0_mag.nii"])
    @test bm == "/r/x_brain_mask.nii" && recon == ["/r/img_Slice0_Rep1_Avg0_Echo0_Phase0_mag.nii"]
    @test split_inputs(["/r/a_brain_mask_edited.nii.gz", "/r/recon"])[1] == "/r/a_brain_mask_edited.nii.gz"
    @test split_inputs(["/r/recon"]) == (nothing, ["/r/recon"])
    @test_throws ErrorException split_inputs(["/r/x_brain_mask.nii"])
    mp = mre_params("s", (64, 64, 32), (2.0, 2.0, 2.5), 50, NaN)
    @test mp.FOVz == 80.0 && mp.nx == 64.0 && mp.freq == 50.0
    js = JSON3.read(JSON3.write(map(_json_safe, mp)))
    @test js.oss_snr === nothing
    @test collect(keys(js))[1:3] == [:subj, :FOVx, :FOVy]
    @test param(Dict{String,Any}("a" => ""), "a", 1) == 1
    @test length(_descrip("MRE wave displacement")) == 80
end

@testset "bridge parses the worker protocol" begin
    push!(LOAD_PATH, joinpath(@__DIR__, "..", "..", "..", "julia", "ForgeStudio"))
    m = Module(:ExtractWavesBridge)
    Core.eval(m, :(using ForgeStudio))        # as forge_dispatcher.jl does
    Base.include(m, joinpath(@__DIR__, "extract_waves.jl"))
    events = Tuple{Symbol,Any}[]
    sink = (k, p) -> push!(events, (k, p))
    outs = String[]
    m.handle_worker_line("WAVES_PROGRESS:{\"step\":4,\"total\":7,\"label\":\"Phase unwrapping\"}", sink)
    m.handle_worker_line("extract_waves: hello", sink)
    m.handle_worker_line("WAVES_PROGRESS:{\"step\":", sink)
    m.handle_worker_line("   ", sink)
    m.handle_worker_line("WAVES_DONE:{\"outputs\":[\"a.nii\",\"b.mat\"]}", sink; outputs = outs)
    @test events[1] == (:progress, (4, 7, "Phase unwrapping"))
    @test events[2] == (:log, "extract_waves: hello")
    @test events[3][1] == :log
    @test length(events) == 3
    @test outs == ["a.nii", "b.mat"]
    withenv("FORGE_SENTINEL_PROJECT" => "/opt/sentinel", "FORGE_SENTINEL_SYSIMAGE" => "/opt/img.so") do
        cmd = m.worker_command("/tmp/cfg.json")
        @test "--project=/opt/sentinel" in cmd.exec
        @test "/opt/img.so" in cmd.exec
        @test cmd.exec[end] == "/tmp/cfg.json"
    end
end
