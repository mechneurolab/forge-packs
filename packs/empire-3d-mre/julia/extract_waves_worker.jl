#!/usr/bin/env julia
# extract_waves_worker.jl
#
# EMPIRE harmonic wave extraction worker. Spawned by extract_waves.jl (the
# FORGE Studio module) in the bundled Sentinel Julia project, which provides
# FFTW, ROMEO, MAT, NIfTI and JSON3; the Studio dispatcher's own project has only JSON3.
#
# Usage:
#   julia --project=<sentinel-project> extract_waves_worker.jl <config.json>
#
# Protocol (stdout, flushed each line):
#   WAVES_PROGRESS:{"step":N,"total":7,"label":"..."}
#   WAVES_DONE:{"outputs":[...]}
# Any other stdout line is a log message. Warnings go to stderr.
#
# Inputs: the forgeSense output directory (or a file in it) holding per-rep
# img_Slice<s>_Rep<r>_Avg0_Echo0_Phase0_{mag,phs}.nii pairs, plus the approved
# *_brain_mask*.nii[.gz] from the Brain Masking stage when it runs first (it
# passes its inputs through, so the inputs are [brain mask, recon outputs...]).
# Outputs (same names and layout as the former extractWaves.m):
#   <base>_wave_{Z,Y,X}_{real,imag}.nii   (displacement_component = "all")
#   <base>_mre_params.json, <base>_mask.nii, <base>_t2stack.nii,
#   <base>_mre_sentinel.mat (Sentinel read_mre_uiuc layout, MAT v7.3)
# where <base> is the recon directory name.

using JSON3

include(joinpath(@__DIR__, "wave_core.jl"))

const TOTAL_STEPS = 7

function emit(tag::AbstractString, payload)
    println(stdout, tag * ":" * JSON3.write(payload))
    flush(stdout)
    return nothing
end

progress(step, label) = emit("WAVES_PROGRESS", (step = step, total = TOTAL_STEPS, label = label))
say(msg) = (println(stdout, "extract_waves: " * msg); flush(stdout))
warn(msg) = (println(stderr, "extract_waves: warning: " * msg); flush(stderr))

# Parameter with default; empty strings/vectors count as unset (forge_param).
function param(params::AbstractDict, name::AbstractString, default)
    v = get(params, name, nothing)
    (v === nothing || (v isa AbstractString && isempty(v)) || (v isa AbstractVector && isempty(v))) &&
        return default
    return v
end

_toint(x::Integer) = Int(x)
_toint(x::Real) = Int(x)
_toint(x::AbstractString) = parse(Int, x)
_tofloat(x::Real) = Float64(x)
_tofloat(x::AbstractString) = parse(Float64, x)

"""
    recon_dir(path) -> String

The pipeline may hand over a file from the recon stage; use its directory.
"""
recon_dir(p::AbstractString) = isfile(p) ? dirname(p) : p

const _BRAIN_MASK_RE = r"_brain_mask[^/\\]*\.nii(\.gz)?$"

"""
    split_inputs(paths) -> (brain_mask_path or nothing, recon_paths)

Separate the Brain Masking stage's mask from the recon outputs.
"""
function split_inputs(paths)
    ps = String.(collect(paths))
    is_mask = [occursin(_BRAIN_MASK_RE, p) for p in ps]
    recon = ps[.!is_mask]
    isempty(recon) && error("no reconstruction inputs given")
    return (any(is_mask) ? ps[findfirst(is_mask)] : nothing), recon
end

"""
    subject_name(dir) -> String

MATLAB `fileparts(dir)` name: the last path component minus any extension.
"""
subject_name(dir::AbstractString) = first(splitext(basename(rstrip(dir, ['/', '\\']))))

const _MAG_RE = r"^img_Slice(\d+)_Rep(\d+)_.*_mag\.nii$"

"""
    scan_recon_files(names) -> (slices, reps)

Sorted unique slice and rep indices from the recon file names.
"""
function scan_recon_files(names)
    slices = Int[]
    reps = Int[]
    for n in names
        m = match(_MAG_RE, n)
        m === nothing && continue
        push!(slices, parse(Int, m.captures[1]))
        push!(reps, parse(Int, m.captures[2]))
    end
    return sort(unique(slices)), sort(unique(reps))
end

mag_name(s, r) = "img_Slice$(s)_Rep$(r)_Avg0_Echo0_Phase0_mag.nii"
phs_name(s, r) = "img_Slice$(s)_Rep$(r)_Avg0_Echo0_Phase0_phs.nii"

# NIfTI descrip field: fixed 80-byte, NUL-padded.
_descrip(s::AbstractString) = (b = codeunits(s); ntuple(i -> i <= length(b) ? b[i] : 0x00, 80))

"""
    mre_params(subj, dims, voxel, freq, snr) -> NamedTuple

The `mreParams` struct extractWaves.m saved, in the same field order.
"""
function mre_params(subj, dims, voxel, freq, snr)
    nx, ny, nz = dims
    dx, dy, dz = voxel
    return (subj = String(subj),
            FOVx = nx * dx, FOVy = ny * dy, FOVz = nz * dz,
            nx = Float64(nx), ny = Float64(ny), nz = Float64(nz),
            dx = dx, dy = dy, dz = dz,
            freq = Float64(freq), oss_snr = Float64(snr))
end

# JSON cannot hold NaN/Inf; MATLAB's jsonencode writes them as null.
_json_safe(x::AbstractFloat) = isfinite(x) ? x : nothing
_json_safe(x) = x

function main(args = ARGS)
    isempty(args) && error("usage: extract_waves_worker.jl <config.json>")
    cfg = JSON3.read(read(args[1], String), Dict{String,Any})
    output_dir = String(cfg["output_dir"])
    inputs = cfg["input_paths"]
    isempty(inputs) && error("no input paths in step config")
    brain_mask_path, recon_inputs = split_inputs(inputs)
    rdir = recon_dir(first(recon_inputs))
    params = Dict{String,Any}(String(k) => v for (k, v) in pairs(get(cfg, "params", Dict{String,Any}())))

    freq_hz    = _tofloat(param(params, "frequency_hz", 50.0))
    num_harm   = _toint(param(params, "num_harmonics", 1))
    disp_comp  = String(param(params, "displacement_component", "all"))
    n_echoes   = _toint(param(params, "n_echoes", 2))
    n_meg      = _toint(param(params, "n_meg_directions", 3))
    disp_scale = _tofloat(param(params, "phase_to_disp_scale", 1.764))
    meg_order  = _toint.(collect(param(params, "meg_order", [3, 1, 2])))

    disp_comp in ("all", "magnitude", "x", "y", "z") ||
        error("Unknown displacement_component: $disp_comp")
    say("recon_dir=$rdir freq=$(freq_hz) Hz harmonics=$num_harm")
    say("n_echoes=$n_echoes n_meg=$n_meg scale=$disp_scale")
    num_harm == 1 || warn("only the first harmonic is written (num_harmonics=$num_harm), as in extractWaves.m")

    @eval Main using NIfTI, MAT
    Base.invokelatest(_run, rdir, brain_mask_path, output_dir, freq_hz, disp_comp, n_echoes,
                      n_meg, disp_scale, meg_order)
end

function _run(rdir, brain_mask_path, output_dir, freq_hz, disp_comp, n_echoes, n_meg,
              disp_scale, meg_order)
    NIfTI = Main.NIfTI
    MAT = Main.MAT
    mkpath(output_dir)

    # ── Step 1: load the mag/phase pairs ────────────────────────────────────
    progress(1, "Loading images")
    slices, reps = scan_recon_files(readdir(rdir))
    isempty(reps) && error("No magnitude NIfTIs found in $rdir")
    length(slices) == 1 || error("expected a single 3D slab, found $(length(slices)) slices in $rdir")
    s = only(slices)
    n_reps = length(reps)
    n_reps % (n_echoes * n_meg) == 0 || error(
        "n_reps ($n_reps) is not divisible by n_echoes*n_meg ($n_echoes*$n_meg=$(n_echoes * n_meg))")
    say("found 1 slice(s), $n_reps rep(s) → $n_echoes echoes x $n_meg MEG x " *
        "$(n_reps ÷ (n_echoes * n_meg)) phases")

    ref = NIfTI.niread(joinpath(rdir, mag_name(s, first(reps))))
    vol = size(ref.raw)
    length(vol) >= 3 || error("expected 3D recon volumes, got size $vol")
    nx, ny, nz = vol[1], vol[2], vol[3]
    px = ref.header.pixdim
    voxel = (Float64(px[2]), Float64(px[3]), Float64(px[4]))
    say("volume size = [$nx $ny $nz], voxel = [$(voxel[1]) $(voxel[2]) $(voxel[3])] mm")

    img = Array{ComplexF32,4}(undef, nx, ny, nz, n_reps)
    for (ri, r) in enumerate(reps)
        mp = joinpath(rdir, mag_name(s, r))
        pp = joinpath(rdir, phs_name(s, r))
        isfile(mp) || error("Missing: $mp")
        isfile(pp) || error("Missing: $pp")
        mag = Float32.(NIfTI.niread(mp).raw)
        phs = Float32.(NIfTI.niread(pp).raw)
        img[:, :, :, ri] .= reshape(mag, nx, ny, nz) .* cis.(reshape(phs, nx, ny, nz))
        ri % 10 == 0 && progress(1, "Loading $ri/$n_reps")
    end
    say("loaded $n_reps file pairs")

    brain_mask = nothing
    if brain_mask_path === nothing
        say("no brain mask input; using a magnitude threshold")
    else
        raw_mask = NIfTI.niread(brain_mask_path).raw
        length(raw_mask) == nx * ny * nz || error(
            "brain mask $brain_mask_path is $(size(raw_mask)) but the reconstruction is $nx x $ny x $nz")
        brain_mask = reshape(raw_mask .> 0, nx, ny, nz)
        say("using brain mask $brain_mask_path ($(count(brain_mask)) voxels)")
    end

    # ── Steps 2–6: the numerics ─────────────────────────────────────────────
    progress(2, "Reshaping data")
    res = extract_waves(img, voxel; brain_mask = brain_mask, n_echoes = n_echoes, n_meg = n_meg,
                        disp_scale = disp_scale, meg_order = meg_order,
                        on_step = progress, on_warn = warn)
    X, Y, Z = res.Xmotion, res.Ymotion, res.Zmotion
    rng(a) = "[$(minimum(abs, a)), $(maximum(abs, a))]"
    say("wave field range — X:$(rng(X)) Y:$(rng(Y)) Z:$(rng(Z))")
    say("OSS-SNR = $(res.oss_snr)")

    # ── Outputs ─────────────────────────────────────────────────────────────
    base = subject_name(rdir)
    out(name) = joinpath(output_dir, "$(base)_$(name)")

    # Every NIfTI reuses the first recon volume's header (geometry, scaling),
    # rewritten as a 3D float32 volume, like extractWaves' save_untouch_nii.
    function write_scalar(path, data)
        hdr = deepcopy(ref.header)
        hdr.descrip = _descrip("MRE wave displacement")
        NIfTI.niwrite(path, NIfTI.NIVolume(hdr, Float32.(data)))
        return path
    end

    mp = mre_params(base, (nx, ny, nz), voxel, freq_hz, res.oss_snr)
    json_path = out("mre_params.json")
    open(json_path, "w") do io
        JSON3.write(io, map(_json_safe, mp))
    end
    say("saved MRE params → $json_path")

    mask_path = write_scalar(out("mask.nii"), res.mask)
    t2_path = write_scalar(out("t2stack.nii"), res.t2stack)

    # Sentinel handoff in the read_mre_uiuc layout.
    mat_path = out("mre_sentinel.mat")
    MAT.matwrite(mat_path, Dict{String,Any}(
        "Xmotion" => X, "Ymotion" => Y, "Zmotion" => Z,
        "t2stack" => res.t2stack,
        "mask" => Array{Bool}(res.mask),
        "mreParams" => Dict{String,Any}(String(k) => v for (k, v) in pairs(mp)),
    ); compress = false)
    say("wrote Sentinel handoff: $mat_path")

    progress(7, "Writing output")
    outputs = String[]
    complex_pair(comp, data) = append!(outputs, [write_scalar(out("wave_$(comp)_real.nii"), real.(data)),
                                                 write_scalar(out("wave_$(comp)_imag.nii"), imag.(data))])
    if disp_comp == "all"
        complex_pair("Z", Z); complex_pair("Y", Y); complex_pair("X", X)
    elseif disp_comp == "magnitude"
        push!(outputs, write_scalar(out("wave_mag.nii"), sqrt.(abs2.(X) .+ abs2.(Y) .+ abs2.(Z))))
    elseif disp_comp == "x"
        complex_pair("X", X)
    elseif disp_comp == "y"
        complex_pair("Y", Y)
    else
        complex_pair("Z", Z)
    end
    append!(outputs, [json_path, mask_path, t2_path, mat_path])
    say("complete — $(length(outputs)) output file(s)")
    emit("WAVES_DONE", (outputs = outputs,))
    return outputs
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
