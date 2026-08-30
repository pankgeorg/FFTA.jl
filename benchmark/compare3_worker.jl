#!/usr/bin/env julia
# One implementation of the `compare3.jl` sweep, run in its own process and
# environment (so that FFTW.jl and FFTA.jl never share a process and dispatch
# is whatever the user of that package would get).
#
#     julia -t N --project=ENV compare3_worker.jl --impl fftw|ffta --out FILE.json [case options]
#
# Case options are those of `compare3.jl` (--only, --kinds, --maxlog2, --sizes,
# --seconds, --no-measure). Julia's thread count (-t) is the plan thread count.
using Statistics, LinearAlgebra, Random, Dates
using JSON
include(joinpath(@__DIR__, "cases.jl"))

function getopt(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
const IMPL     = getopt("--impl", "ffta")
const OUTFILE  = getopt("--out", "compare3_$(IMPL).json")
const ONLY     = split(getopt("--only", "1d,nd,batched,threads"), ",")
const KINDS    = Tuple(Symbol.(split(getopt("--kinds", "fft,rfft"), ",")))
const SIZES    = let v = getopt("--sizes", ""); isempty(v) ? Int[] : parse.(Int, split(v, ",")) end
const MAXLOG2  = parse(Int, getopt("--maxlog2", "22"))
const SECONDS  = parse(Float64, getopt("--seconds", "0.5"))
const MEASURE  = !("--no-measure" in ARGS)
const NTHREADS = Threads.nthreads()

if IMPL == "fftw"
    using FFTW
    FFTW.set_num_threads(NTHREADS)
    mkplan(kind, x, dims; measure) = kind === :fft ?
        plan_fft(x, dims; flags = measure ? FFTW.MEASURE : FFTW.ESTIMATE) :
        plan_rfft(x, dims; flags = measure ? FFTW.MEASURE : FFTW.ESTIMATE)
    const VERSION_STR = "FFTW.jl $(pkgversion(FFTW)) / FFTW $(FFTW.version) ($(FFTW.get_provider()))"
else
    using FFTA
    # `num_threads` is the plan keyword of the threaded FFTA; older versions
    # accept and ignore unknown keywords.
    function mkplan(kind, x, dims; measure)
        try
            kind === :fft ? plan_fft(x, dims; num_threads = NTHREADS) : plan_rfft(x, dims; num_threads = NTHREADS)
        catch err
            # FFTA 0.3.1 accepts a vector region but not a tuple
            (err isa MethodError && dims isa Tuple) || rethrow()
            kind === :fft ? plan_fft(x, collect(dims); num_threads = NTHREADS) : plan_rfft(x, collect(dims); num_threads = NTHREADS)
        end
    end
    const VERSION_STR = let sha = try
            strip(read(`git -C $(pkgdir(FFTA)) rev-parse --short HEAD`, String))
        catch
            ""
        end
        "FFTA.jl $(pkgversion(FFTA))" * (isempty(sha) ? "" : " @ $sha")
    end
end

function timeit(f; seconds = SECONDS)
    f(); f()
    t0 = time_ns(); f(); t1 = time_ns()
    evals = max(1, ceil(Int, 20_000 / max(t1 - t0, 1)))
    times = Float64[]
    deadline = time_ns() + seconds * 1e9
    while time_ns() < deadline || length(times) < 5
        ts = time_ns()
        for _ in 1:evals
            f()
        end
        push!(times, (time_ns() - ts) / evals)
        length(times) >= 100_000 && break
    end
    return (min = minimum(times) * 1e-9, median = median(times) * 1e-9, samples = length(times))
end

# deterministic input so that every implementation transforms the same data
function input(c)
    rng = Xoshiro(0x5eed)
    c.kind === :fft ? randn(rng, Complex{c.T}, c.sz) : randn(rng, c.T, c.sz)
end

const RESULTS = Dict{String,Any}[]
const META = Dict("impl" => IMPL, "version" => VERSION_STR, "julia" => string(VERSION),
                  "threads" => NTHREADS, "arch" => String(Sys.ARCH), "cpu" => Sys.CPU_NAME,
                  "date" => string(now(UTC)), "seconds" => SECONDS)
function flush_results(complete)
    open(OUTFILE, "w") do io
        JSON.print(io, Dict("meta" => META, "complete" => complete, "results" => RESULTS), 1)
    end
end

cases = case_list(; only = ONLY, kinds = KINDS, maxlog2 = MAXLOG2, sizes = SIZES,
                    nthreads = NTHREADS, measure = MEASURE && IMPL == "fftw")
println("== compare3 worker: $VERSION_STR, $(length(cases)) cases, $NTHREADS threads")
for c in cases
    entry = Dict{String,Any}("key" => casekey(c), "kind" => String(c.kind), "T" => string(c.T),
                             "size" => collect(c.sz), "dims" => collect(c.dims), "shape" => c.shape,
                             "class" => c.class, "nthreads" => c.nthreads, "measure" => c.measure,
                             "n" => prod(c.sz[d] for d in c.dims))
    x = input(c)
    try
        p = mkplan(c.kind, x, c.dims; measure = c.measure)
        y = p * x
        # checksum (compared against the reference implementation by the driver)
        entry["norm"] = norm(y); entry["sum_re"] = real(sum(y)); entry["sum_im"] = imag(sum(y))
        c.measure || (entry["plan_min"] = timeit(() -> mkplan(c.kind, x, c.dims; measure = false)).min)
        has_mul = try
            mul!(y, p, x); true
        catch err
            err isa MethodError || rethrow(); false
        end
        if has_mul
            entry["exec_min"] = timeit(() -> mul!(y, p, x)).min
            entry["exec_alloc"] = @allocated mul!(y, p, x)
            entry["api"] = "mul!"
        else
            entry["exec_min"] = timeit(() -> p * x).min
            entry["exec_alloc"] = @allocated p * x
            entry["api"] = "*"
        end
    catch err
        entry["error"] = sprint(showerror, err)[1:min(end, 200)]
    end
    push!(RESULTS, entry); flush_results(false)
    println(rpad(entry["key"], 52), haskey(entry, "exec_min") ?
            string(round(entry["exec_min"] * 1e6; digits = 2), " us") : "✗ " * entry["error"][1:min(end, 60)])
    flush(stdout)
end
flush_results(true)
println("wrote $(length(RESULTS)) results to $OUTFILE")
