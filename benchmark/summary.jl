#!/usr/bin/env julia
#=
Render a short RESULTS.md from two `compare3.jl` columns: FFTW and FFTA.

    julia --project=. summary.jl DIR [--threads N] [--out RESULTS.md]

`DIR` holds `fftw_tN.json` and `ffta_tN.json` (the FFTA column must be
named `ffta`). Prints (1) absolute execution times for a fixed set of
representative cases, (2) FFTA/FFTW geometric means per size class, (3) the
share of cases where FFTA is faster. Nothing else.
=#
import Pkg; Pkg.instantiate()
using JSON, Printf, Statistics

function getopt(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
const DIR = first(filter(a -> !startswith(a, "--") && !(a in (getopt("--threads", ""), getopt("--out", ""))), ARGS))
const NT = parse(Int, getopt("--threads", "1"))
const OUT = getopt("--out", joinpath(DIR, "RESULTS.md"))
load(f) = Dict(e["key"] => e for e in JSON.parsefile(f)["results"])
fftw = load(joinpath(DIR, "fftw_t$(NT).json"))
ffta = load(joinpath(DIR, "ffta_t$(NT).json"))
meta = JSON.parsefile(joinpath(DIR, "ffta_t$(NT).json"))["meta"]
fmt(t) = t < 1e-6 ? @sprintf("%.0f ns", t * 1e9) : t < 1e-3 ? @sprintf("%.1f µs", t * 1e6) : @sprintf("%.2f ms", t * 1e3)
suffix = NT > 1 ? " ($(NT) thr)" : ""
t(d, k) = (e = get(d, k * suffix, nothing); e === nothing || !haskey(e, "exec_min") ? nothing : e["exec_min"])

representative = [
    ("fft ComplexF64 2^10",  "fft Float64 1024 dims=1"), ("fft ComplexF64 2^16", "fft Float64 65536 dims=1"),
    ("fft ComplexF64 2^20",  "fft Float64 1048576 dims=1"), ("fft ComplexF32 2^20", "fft Float32 1048576 dims=1"),
    ("fft ComplexF64 1000 (2³·5³)", "fft Float64 1000 dims=1"), ("fft ComplexF64 10⁶", "fft Float64 1000000 dims=1"),
    ("fft ComplexF64 49757 (prime)", "fft Float64 49757 dims=1"), ("fft ComplexF64 293201 (prime)", "fft Float64 293201 dims=1"), ("fft ComplexF64 12297 (3·4099)", "fft Float64 12297 dims=1"),
    ("rfft Float64 2^12", "rfft Float64 4096 dims=1"), ("rfft Float64 2^16", "rfft Float64 65536 dims=1"),
    ("rfft Float64 2^20", "rfft Float64 1048576 dims=1"), ("rfft Float32 2^16", "rfft Float32 65536 dims=1"),
    ("fft ComplexF64 256×256 (2D)", "fft Float64 256×256 dims=1,2"), ("fft ComplexF64 1024×1024 (2D)", "fft Float64 1024×1024 dims=1,2"),
    ("rfft Float64 1024×1024 (2D)", "rfft Float64 1024×1024 dims=1,2"), ("fft ComplexF64 64³ (3D)", "fft Float64 64×64×64 dims=1,2,3"),
    ("fft ComplexF64 4096×64 along dim 1", "fft Float64 4096×64 dims=1"), ("rfft Float64 4096×64 along dim 1", "rfft Float64 4096×64 dims=1"),
    ("fft ComplexF64 64×4096 along dim 2", "fft Float64 64×4096 dims=2"), ("rfft Float64 1024×1024 along dim 1", "rfft Float64 1024×1024 dims=1"),
]
open(OUT, "w") do io
    println(io, "# FFTA vs FFTW — execution time\n")
    println(io, "$(meta["version"]) vs $(JSON.parsefile(joinpath(DIR, "fftw_t$(NT).json"))["meta"]["version"]), $(meta["cpu"]) ($(meta["arch"])), Julia $(meta["julia"]), $(NT) thread$(NT > 1 ? "s" : ""), $(meta["date"][1:10]). ",
                "Planned execution (`mul!`), minimum over samples; FFTW with its default `ESTIMATE` plans. Produced by `benchmark/summary.jl` from `compare3.jl` output.\n")
    println(io, "## Representative cases\n")
    println(io, "| case | FFTW | FFTA | FFTA / FFTW |\n|:--|--:|--:|--:|")
    for (label, key) in representative
        a = t(fftw, key); b = t(ffta, key)
        (a === nothing || b === nothing) && continue
        @printf(io, "| %s | %s | **%s** | %.2f× |\n", label, fmt(a), fmt(b), b / a)
    end
    # class geomeans
    println(io, "\n## By size class (geometric mean of FFTA / FFTW, lower is better)\n")
    groups = Dict{Tuple{String,String},Vector{Float64}}()
    nfaster = 0; ntotal = 0
    for (k, e) in ffta
        haskey(e, "exec_min") && haskey(fftw, k) && haskey(fftw[k], "exec_min") || continue
        e["measure"] && continue
        r = e["exec_min"] / fftw[k]["exec_min"]
        cls = e["shape"] == "1d" ? e["class"] : replace(e["shape"], "batched_dim1" => "batched, along dim 1", "batched_dim2" => "batched, along dim 2")
        lab = (e["kind"] == "fft" ? "Complex" * replace(e["T"], "Float" => "F") : e["T"]) * " " * e["kind"]
        push!(get!(groups, (cls, lab), Float64[]), r)
        ntotal += 1; r < 1 && (nfaster += 1)
    end
    labs = ["ComplexF64 fft", "ComplexF32 fft", "Float64 rfft", "Float32 rfft"]
    println(io, "| class | " * join(labs, " | ") * " |\n|:--|" * "--:|"^length(labs))
    for cls in ["pow2", "smooth", "prime", "awkward", "2d", "3d", "batched, along dim 1", "batched, along dim 2"]
        row = "| $cls |"
        for lab in labs
            v = get(groups, (cls, lab), Float64[])
            row *= isempty(v) ? " — |" : @sprintf(" %.2f× (%d) |", exp(mean(log.(v))), length(v))
        end
        println(io, row)
    end
    all_r = vcat(values(groups)...)
    @printf(io, "\n**All %d cases: %.2f× geometric mean; FFTA faster than FFTW in %d of them.** ", ntotal, exp(mean(log.(all_r))), nfaster)
    println(io, "Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.")
end
println("wrote $OUT")
