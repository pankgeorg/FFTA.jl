#!/usr/bin/env julia
#=
The FFT calls DSP.jl's `conv` makes, replayed on FFTA and on FFTW.

DSP.jl's test suite runs 1.8× longer on FFTA than on FFTW, all of it in
`test/dsp.jl`. This script isolates that: it performs the same sequences of
AbstractFFTs calls as DSP's convolution kernels — one-shot, planning every
time, as DSP does — over the size and element-type patterns of that test
file, and reports for every case the time on FFTA and on FFTW, and how much
of FFTA's time is plan construction. No DSP.jl dependency (its size rules,
`nextfastfft` and `optimalfftfiltlength`, are copied below).

    julia --project=. dsp_cases.jl [--quick] [--ffta PATH] [--seconds S]

Runs each implementation in its own environment/process (`envs/fftw`,
`envs/ffta` → this checkout unless `--ffta PATH`), like compare3.jl, then
prints the table. `--quick` skips the 3-D reference convolutions of 256³
elements. Iterate: change FFTA, rerun (`envs/ffta` develops the checkout, so
no reinstall is needed), compare the group totals.

Groups (mirroring `test/dsp.jl`):
  conv-1D simple      `_conv_kern_fft!`: plan_rfft + 2 executions + one-shot irfft
                      (complex: plan_fft! + inv + 3 executions), n = nextfastfft(M+N-1)
  conv-1D overlapsave `unsafe_conv_kern_os!`: plan_rfft + plan_brfft once, then
                      2 mul! per block, nfft = optimalfftfiltlength
  conv-2D / conv-ND   the same on the 2-D and small N-d shapes of the tests
  overlap-save tests  128^N ⋆ nsmall^N, N = 1..3, three element types: the
                      overlap-save run and its full-size reference convolution
=#
using LinearAlgebra, Statistics, Printf

function getopt(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
const QUICK = "--quick" in ARGS
const SECONDS = parse(Float64, getopt("--seconds", "0.3"))

# ---------------------------------------------------------------------------
# DSP.jl's size rules (src/util.jl, src/dspbase.jl)
# ---------------------------------------------------------------------------
nextfastfft(n::Integer) = nextprod((2, 3, 5, 7), n)
nextfastfft(ns::Tuple) = nextfastfft.(ns)
os_fft_complexity(nfft, nb) = (nfft * log2(nfft) + nfft) / (nfft - nb + 1)
function optimalfftfiltlength(nb, nx)
    nfull = nb + nx - 1
    first_pow2 = ceil(Int, log2(nb))
    max_pow2 = ceil(Int, log2(nfull))
    prev = os_fft_complexity(2^first_pow2, nb)
    pow2 = first_pow2 + 1
    while pow2 <= max_pow2
        c = os_fft_complexity(2^pow2, nb)
        c > prev && break
        prev = c
        pow2 += 1
    end
    nfft = pow2 > max_pow2 ? 2^max_pow2 : 2^(pow2 - 1)
    nfft > nfull ? nextfastfft(nfull) : nfft
end

# ---------------------------------------------------------------------------
# worker: the calls DSP makes
# ---------------------------------------------------------------------------
if "--worker" in ARGS
    using JSON
    const IMPL = getopt("--worker", "ffta")
    if IMPL == "fftw"
        using FFTW
    else
        using FFTA
    end

    # returns (first call in this process — includes compilation of anything
    # this case's types and shapes need —, minimum of the repeats after it)
    function timeit(f; seconds = SECONDS)
        first = @elapsed f()
        best = Inf; deadline = time_ns() + seconds * 1e9; n = 0
        while time_ns() < deadline || n < 3
            t = @elapsed f(); best = min(best, t); n += 1
        end
        (first, best)
    end

    # `_conv_kern_fft!` for real input: plan once, two executions, one-shot irfft
    function conv_simple(::Type{T}, sout::Tuple) where {T<:Real}
        nffts = nextfastfft(sout)
        padded = zeros(T, nffts)
        uf0 = rfft(padded)
        work() = begin
            p = plan_rfft(padded)
            uf = p * padded
            vf = p * padded
            uf .*= vf
            irfft(uf, nffts[1])
        end
        planonly() = (plan_rfft(padded); plan_irfft(uf0, nffts[1]))
        return work, planonly
    end
    # `_conv_kern_fft!` for complex input: in-place plan and its inverse, three executions
    function conv_simple(::Type{T}, sout::Tuple) where {T<:Complex}
        nffts = nextfastfft(sout)
        upad = zeros(T, nffts); vpad = zeros(T, nffts)
        work() = begin
            p! = plan_fft!(upad)
            ip! = inv(p!)
            p! * upad
            p! * vpad
            upad .*= vpad
            ip! * upad
        end
        planonly() = (p! = plan_fft!(upad); inv(p!))
        return work, planonly
    end
    # `unsafe_conv_kern_os!`: plans once, then per block one forward and one backward
    function conv_os(::Type{T}, su::Tuple, sv::Tuple, nfft::Int) where {T<:Real}
        N = length(su)
        nffts = ntuple(_ -> nfft, N)
        nblocks = prod(cld.(su .+ sv .- 1, nffts .- sv .+ 1))
        tdbuff = zeros(T, nffts)
        fdbuff = zeros(Complex{T}, ntuple(i -> i == 1 ? nffts[i] >> 1 + 1 : nffts[i], N))
        filt = zeros(Complex{T}, size(fdbuff))
        work() = begin
            p = plan_rfft(tdbuff); ip = plan_brfft(fdbuff, nffts[1])
            mul!(filt, p, tdbuff)                 # the filter's transform
            for _ in 1:nblocks
                mul!(fdbuff, p, tdbuff); fdbuff .*= filt; mul!(tdbuff, ip, fdbuff)
            end
        end
        planonly() = (plan_rfft(tdbuff); plan_brfft(fdbuff, nffts[1]))
        return work, planonly, nblocks
    end
    function conv_os(::Type{T}, su::Tuple, sv::Tuple, nfft::Int) where {T<:Complex}
        N = length(su)
        nffts = ntuple(_ -> nfft, N)
        nblocks = prod(cld.(su .+ sv .- 1, nffts .- sv .+ 1))
        buff = zeros(T, nffts); filt = zeros(T, nffts)
        work() = begin
            p = plan_fft!(buff); ip = inv(p).p
            p * buff; copyto!(filt, buff)         # the filter's transform, in place
            for _ in 1:nblocks
                p * buff; buff .*= filt; ip * buff
            end
        end
        planonly() = (p = plan_fft!(buff); inv(p).p)
        return work, planonly, nblocks
    end

    results = Dict{String,Any}[]
    function record!(group, case, work, planonly; extra = Dict())
        first, t = timeit(work); _, tp = timeit(planonly)
        push!(results, merge(Dict("group" => group, "case" => case, "total" => t, "plans" => tp, "first" => first), extra))
        @printf("%-22s %-46s %9.3f ms  (plans %5.0f%%, first call %8.1f ms)\n", group, case, t * 1e3, 100tp / t, first * 1e3); flush(stdout)
    end
    function all_cases!()
        tname(T) = T === Float64 ? "Float64" : T === Float32 ? "Float32" : "ComplexF64"

        # conv-1D and xcorr: M, N ∈ {10, 200}, Float64 and ComplexF64, both FFT algorithms
        for M in (10, 200), N in (10, 200), T in (Float64, ComplexF64)
            work, planonly = conv_simple(T, (M + N - 1,))
            record!("conv-1D simple", "$(tname(T)) $M⋆$N → n=$(nextfastfft(M + N - 1))", work, planonly)
            nb, nx = minmax(M, N)
            nfft = optimalfftfiltlength(nb, nx)
            work, planonly, nbl = conv_os(T, (nx,), (nb,), nfft)
            record!("conv-1D overlapsave", "$(tname(T)) $M⋆$N → nfft=$nfft, $nbl blocks", work, planonly)
        end
        # conv-2D
        for (M1, M2) in ((10, 20), (190, 200)), (N1, N2) in ((20, 10), (210, 200)), T in (Float64, ComplexF64)
            sout = (M1 + N1 - 1, M2 + N2 - 1)
            work, planonly = conv_simple(T, sout)
            record!("conv-2D simple", "$(tname(T)) $(M1)×$(M2) ⋆ $(N1)×$(N2) → $(join(nextfastfft(sout), "×"))", work, planonly)
            nb = min(M1 * M2, N1 * N2) == M1 * M2 ? (M1, M2) : (N1, N2)
            nx = nb == (M1, M2) ? (N1, N2) : (M1, M2)
            nfft = optimalfftfiltlength(maximum(nb), maximum(nx))
            work, planonly, nbl = conv_os(T, nx, nb, nfft)
            record!("conv-2D overlapsave", "$(tname(T)) $(M1)×$(M2) ⋆ $(N1)×$(N2) → nfft=$nfft, $nbl blocks", work, planonly)
        end
        # small N-d convolutions of the "conv" / "conv-ND" testsets
        for (sa, sb) in (((5,), (5,)), ((5, 5), (5, 5)), ((5, 5, 5), (5, 5, 5)), ((3, 3, 6), (2, 2, 2)), ((2, 2, 2, 2, 2, 2), (1, 1, 1, 1, 1, 1)), ((4, 7, 1), (3, 3, 3)))
            work, planonly = conv_simple(Float64, sa .+ sb .- 1)
            record!("conv-ND small", "Float64 $(join(sa, "×")) ⋆ $(join(sb, "×")) → $(join(nextfastfft(sa .+ sb .- 1), "×"))", work, planonly)
        end
        # overlap-save tests: 128^N ⋆ nsmall^N, the overlap-save run and its reference
        for numdim in 1:3, T in (Float32, Float64, ComplexF64), nsmall in (12, 128)
            QUICK && numdim == 3 && nsmall == 128 && continue
            nlarge = 128
            nfft = optimalfftfiltlength(nsmall, nlarge)
            su = ntuple(_ -> nlarge, numdim); sv = ntuple(_ -> nsmall, numdim)
            work, planonly, nbl = conv_os(T, su, sv, nfft)
            record!("os-test overlapsave", "$(tname(T)) $(nlarge)^$(numdim) ⋆ $(nsmall)^$(numdim), nfft=$nfft, $nbl blocks", work, planonly)
            work, planonly = conv_simple(T, su .+ sv .- 1)
            record!("os-test reference", "$(tname(T)) $(nlarge)^$(numdim) ⋆ $(nsmall)^$(numdim) → $(nextfastfft(nlarge + nsmall - 1))^$(numdim)", work, planonly)
        end
        for (nsmall, nfft) in ((12, 256), (13, 32), (12, 32))
            work, planonly, nbl = conv_os(Float64, (128,), (nsmall,), nfft)
            record!("os-test adversarial", "Float64 128 ⋆ $nsmall, nfft=$nfft, $nbl blocks", work, planonly)
        end
        work, planonly, nbl = conv_os(Float64, (25,), (4,), 16)
        record!("os-test adversarial", "Float64 25 ⋆ 4, nfft=16, $nbl blocks", work, planonly)

    end
    all_cases!()
    open(getopt("--out", "dsp_cases_$(IMPL).json"), "w") do io
        JSON.print(io, results, 1)
    end
    exit()
end

# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------
import Pkg; Pkg.instantiate()
using JSON
const FFTA_PATH = abspath(getopt("--ffta", joinpath(@__DIR__, "..")))
const OUTDIR = joinpath(@__DIR__, "dsp_cases_results"); mkpath(OUTDIR)
function env(name, spec)
    dir = joinpath(@__DIR__, "envs", name)
    if !isfile(joinpath(dir, "Project.toml"))
        mkpath(dir)
        code = spec == "fftw" ? """Pkg.add(["FFTW", "JSON"])""" : """Pkg.develop(path = "$spec"); Pkg.add("JSON")"""
        run(`$(Base.julia_cmd()) --project=$dir -e "import Pkg; $code; Pkg.precompile()"`)
    end
    dir
end
pass = filter(a -> a in ("--quick",), ARGS); append!(pass, ["--seconds", string(SECONDS)])
for (name, spec) in (("fftw", "fftw"), ("ffta", FFTA_PATH))
    out = joinpath(OUTDIR, "$name.json")
    println("== $name"); flush(stdout)
    run(`$(Base.julia_cmd()) -t 1 --project=$(env(name, spec)) $(@__FILE__) --worker $name --out $out $pass`)
end
fftw = JSON.parsefile(joinpath(OUTDIR, "fftw.json")); ffta = JSON.parsefile(joinpath(OUTDIR, "ffta.json"))
fmt(t) = t < 1e-3 ? @sprintf("%.1f µs", t * 1e6) : t < 1 ? @sprintf("%.2f ms", t * 1e3) : @sprintf("%.2f s", t)
open(joinpath(OUTDIR, "DSP_CASES.md"), "w") do io
    println(io, "# DSP.jl's convolution FFT calls: FFTA vs FFTW\n")
    println(io, "One-shot, planning every time, as DSP.jl does, 1 thread. `steady` = minimum of repeated runs; `first call` = the first run of that case in a fresh process, which includes compiling whatever its element type, shape and size need (a test suite pays this once per combination). `plans` = share of FFTA's steady time spent constructing plans.\n")
    println(io, "| group | case | FFTW steady | FFTA steady | FFTA / FFTW | FFTA plans | FFTW first call | FFTA first call |\n|:--|:--|--:|--:|--:|--:|--:|--:|")
    totals = Dict{String,Vector{Float64}}()
    for (a, b) in zip(fftw, ffta)
        @assert a["case"] == b["case"]
        @printf(io, "| %s | %s | %s | **%s** | %.2f× | %.0f%% | %s | **%s** |\n", a["group"], a["case"], fmt(a["total"]), fmt(b["total"]), b["total"] / a["total"], 100b["plans"] / b["total"], fmt(a["first"]), fmt(b["first"]))
        v = get!(totals, a["group"], zeros(5)); v .+= (a["total"], b["total"], b["plans"], a["first"], b["first"])
    end
    println(io, "\n## Totals per group\n")
    println(io, "| group | FFTW steady | FFTA steady | FFTA / FFTW | FFTW first calls | FFTA first calls | first-call ratio |\n|:--|--:|--:|--:|--:|--:|--:|")
    g = zeros(5)
    for grp in unique(e["group"] for e in fftw)
        v = totals[grp]; g .+= v
        @printf(io, "| %s | %s | **%s** | %.2f× | %s | **%s** | %.2f× |\n", grp, fmt(v[1]), fmt(v[2]), v[2] / v[1], fmt(v[4]), fmt(v[5]), v[5] / v[4])
    end
    @printf(io, "| **all** | %s | **%s** | %.2f× | %s | **%s** | %.2f× |\n", fmt(g[1]), fmt(g[2]), g[2] / g[1], fmt(g[4]), fmt(g[5]), g[5] / g[4])
end
print(read(joinpath(OUTDIR, "DSP_CASES.md"), String))
