# FORGE Studio module for the EMPIRE pack's Wave Extraction step.
#
# The numerics need FFTW, MAT and NIfTI, which live in the bundled Sentinel
# Julia project rather than the dispatcher's. Like sentinel_inversion.jl, this
# module runs a worker (extract_waves_worker.jl) in that project as a
# subprocess and re-emits its WAVES_* stdout protocol through ForgeStudio.
using ForgeStudio
using JSON3

const BAR = "waves"

"""
    handle_worker_line(line, sink; outputs=nothing)

Route one worker stdout line via `sink(kind, payload)`:
`WAVES_PROGRESS:` → `sink(:progress, (step, total, label))`,
`WAVES_DONE:` → appends to `outputs`, anything else non-empty → `sink(:log, line)`.
A malformed protocol line is logged rather than fatal.
"""
function handle_worker_line(line::AbstractString, sink; outputs = nothing)
    if startswith(line, "WAVES_PROGRESS:")
        try
            d = JSON3.read(line[length("WAVES_PROGRESS:")+1:end], Dict)
            sink(:progress, (Int(d["step"]), Int(d["total"]), String(get(d, "label", ""))))
        catch
            sink(:log, String(line))
        end
    elseif startswith(line, "WAVES_DONE:")
        try
            d = JSON3.read(line[length("WAVES_DONE:")+1:end], Dict)
            outputs === nothing || append!(outputs, String.(d["outputs"]))
        catch
            sink(:log, String(line))
        end
    elseif !isempty(strip(line))
        sink(:log, String(line))
    end
    nothing
end

"""
    worker_command(config_path) -> Cmd

The worker runs in the Sentinel project (`FORGE_SENTINEL_PROJECT`, set by the
Studio Julia session) with the bundled sysimage when there is one.
"""
function worker_command(config_path::AbstractString)
    project = get(ENV, "FORGE_SENTINEL_PROJECT", "")
    isempty(project) && error("FORGE_SENTINEL_PROJECT is not set; the Julia session sets it to the bundled Sentinel project")
    sysimg = get(ENV, "FORGE_SENTINEL_SYSIMAGE", "")
    nthreads = get(ENV, "FORGE_WAVES_THREADS", "auto")
    args = String[joinpath(Sys.BINDIR, "julia"), "--project=$project", "--threads=$nthreads"]
    isempty(sysimg) || append!(args, ["--sysimage", sysimg])
    push!(args, joinpath(@__DIR__, "extract_waves_worker.jl"), config_path)
    return Cmd(args)
end

function run(config_path::String)::Vector{String}
    forge_read_config(config_path)
    outputs = String[]
    sink = (kind, payload) -> begin
        if kind === :progress
            forge_progress(BAR, payload[1], payload[2], "Wave Extraction"; postfix = payload[3])
        elseif kind === :log
            forge_log(:info, payload)
        end
    end

    proc = open(worker_command(config_path); read = true)
    try
        for line in eachline(proc)
            handle_worker_line(line, sink; outputs = outputs)
        end
    catch err
        # Cancelling sends SIGINT to this session only; stop the worker too.
        try
            kill(proc)
        catch
        end
        rethrow(err)
    finally
        close(proc)
        wait(proc)
    end

    forge_progress_done(BAR)
    success(proc) || error("wave extraction worker exited with failure (code $(proc.exitcode))")
    isempty(outputs) && error("wave extraction worker produced no outputs")
    return outputs
end
