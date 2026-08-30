#!/usr/bin/env julia
#=
Side-by-side benchmark of several FFT implementations on the `suite.jl` sweep.

Each implementation runs in its own process and environment (FFTW.jl and
FFTA.jl never share a process), then one markdown table is rendered with
execution times, the ratio of each column to the first (FFTW) and the speedup
of each column over a reference column.

    julia --project=. compare3.jl [--impl NAME=SPEC ...] [--ref NAME]
                                  [--threads 1,4,16] [--out DIR] [--table FILE.md]
                                  [--only 1d,nd,batched,threads] [--kinds fft,rfft]
                                  [--classes pow2,smooth,prime,awkward]
                                  [--maxlog2 K] [--seconds S] [--sizes a,b,c]
                                  [--no-measure] [--quick] [--render-only] [--skip-existing]

`SPEC` is `fftw` (FFTW.jl from the registry), `@VERSION` (FFTA from the
registry, e.g. `@0.3.1`), or a path to an FFTA checkout / worktree. The
defaults reproduce the three-way comparison

    --impl fftw=fftw --impl release=@0.3.1 --impl int=..      (this checkout)

and `--ref` names the column that later columns are compared with (default:
the second FFTA column if there is one, else the last). A typical experiment
run is therefore

    julia --project=. compare3.jl --impl fftw=fftw --impl int=/path/to/integration \
                                  --impl new=/path/to/experiment --ref int

Environments are created under `envs/NAME/` (git-ignored) and re-used; delete
one to rebuild it. `--skip-existing` keeps a column whose complete JSON file
is already in `--out` and whose recorded case selection covers the current
one (e.g. to add one column to a finished run); otherwise it is re-measured. Per-run JSON files land in `--out` (default
`compare3_results/`) as `NAME_tN.json`; `--render-only` re-renders the table
from them without benchmarking.
=#
import Pkg; Pkg.instantiate()
using JSON, Printf, Statistics, Dates
include(joinpath(@__DIR__, "cases.jl"))

function getopt(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
getall(flag) = [ARGS[i + 1] for i in findall(==(flag), ARGS)]

const QUICK   = "--quick" in ARGS
const IMPLS   = let v = getall("--impl")
    isempty(v) && (v = ["fftw=fftw", "release=@0.3.1", "int=" * normpath(joinpath(@__DIR__, ".."))])
    [(name = String(first(split(s, "="; limit = 2))), spec = String(last(split(s, "="; limit = 2)))) for s in v]
end
const THREADS = parse.(Int, split(getopt("--threads", "1"), ","))
const OUTDIR  = getopt("--out", joinpath(@__DIR__, "compare3_results"))
const TABLE   = getopt("--table", joinpath(OUTDIR, "COMPARE3.md"))
const REF     = let r = getopt("--ref", "")
    if isempty(r)
        ffta = filter(i -> i.spec != "fftw", IMPLS)
        length(ffta) >= 2 ? ffta[2].name : last(IMPLS).name
    else
        r
    end
end
const ENVDIR  = joinpath(@__DIR__, "envs")
const PASS    = String[]      # case options passed through to the workers
for flag in ("--only", "--kinds", "--sizes", "--classes")
    v = getopt(flag, ""); isempty(v) || append!(PASS, [flag, v])
end
append!(PASS, ["--maxlog2", getopt("--maxlog2", QUICK ? "14" : "22")])
append!(PASS, ["--seconds", getopt("--seconds", QUICK ? "0.1" : "0.5")])
"--no-measure" in ARGS && push!(PASS, "--no-measure")

# ---------------------------------------------------------------------------
# environments
# ---------------------------------------------------------------------------
function ensure_env(impl)
    dir = joinpath(ENVDIR, impl.name)
    isfile(joinpath(dir, "Project.toml")) && return dir
    mkpath(dir)
    @info "creating environment" impl.name impl.spec dir
    code = if impl.spec == "fftw"
        """Pkg.add(["FFTW", "JSON", "Primes"])"""
    elseif startswith(impl.spec, "@")
        """Pkg.add([Pkg.PackageSpec(name = "FFTA", version = "$(impl.spec[2:end])"), Pkg.PackageSpec(name = "JSON"), Pkg.PackageSpec(name = "Primes")])"""
    else
        path = abspath(impl.spec)
        isfile(joinpath(path, "Project.toml")) || error("$(impl.name): no Project.toml in $path")
        """Pkg.develop(path = "$path"); Pkg.add(["JSON", "Primes"])"""
    end
    run(`$(Base.julia_cmd()) --project=$dir -e "import Pkg; $code; Pkg.precompile()"`)
    return dir
end

# Does the case selection recorded in a results file cover the current one?
# (files without a recorded scope — older runs — are accepted with a warning)
function scope_covers(meta)
    haskey(meta, "scope") || (@warn "results file has no recorded scope; assuming it covers this run"; return true)
    sc = meta["scope"]
    cur_only = split(getopt("--only", "1d,nd,batched,threads"), ",")
    cur_kinds = split(getopt("--kinds", "fft,rfft"), ",")
    cur_classes = split(getopt("--classes", "pow2,smooth,prime,awkward"), ",")
    cur_maxlog2 = parse(Int, getopt("--maxlog2", QUICK ? "14" : "22"))
    cur_sizes = let v = getopt("--sizes", ""); isempty(v) ? Int[] : parse.(Int, split(v, ",")) end
    ok = all(o -> o in sc["only"] || (o == "threads"), cur_only) &&   # (the threads section is empty at 1 thread)
         all(k -> k in sc["kinds"], cur_kinds) &&
         (!("1d" in cur_only) || all(c -> c in sc["classes"], cur_classes)) &&
         sc["maxlog2"] >= cur_maxlog2 &&
         (isempty(sc["sizes"]) || (!isempty(cur_sizes) && all(n -> n in sc["sizes"], cur_sizes)))
    return ok
end

function run_worker(impl, nthreads)
    out = joinpath(OUTDIR, "$(impl.name)_t$(nthreads).json")
    if "--skip-existing" in ARGS && isfile(out)
        d = JSON.parsefile(out)
        if get(d, "complete", false) && scope_covers(d["meta"])
            @info "keeping existing results" impl.name nthreads out
            return out
        else
            @warn "existing results do not cover this run (incomplete or narrower scope); re-measuring" impl.name nthreads out
        end
    end
    dir = ensure_env(impl)
    kind = impl.spec == "fftw" ? "fftw" : "ffta"
    cmd = `$(Base.julia_cmd()) -t $nthreads --project=$dir $(joinpath(@__DIR__, "compare3_worker.jl")) --impl $kind --out $out $PASS`
    @info "running" impl.name nthreads
    run(cmd)
    return out
end

mkpath(OUTDIR)
if !("--render-only" in ARGS)
    for nthreads in THREADS, impl in IMPLS
        run_worker(impl, nthreads)
    end
end

# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
fmt_t(t) = t < 1e-6 ? @sprintf("%.0f ns", t * 1e9) : t < 1e-3 ? @sprintf("%.2f µs", t * 1e6) :
           t < 1 ? @sprintf("%.3f ms", t * 1e3) : @sprintf("%.3f s", t)
fmt_r(r) = @sprintf("%.2f×", r)
gmean(v) = exp(mean(log.(v)))
label(e) = (e["kind"] == "fft" ? "Complex" * replace(e["T"], "Float" => "F") : e["T"]) * " " * e["kind"]

loaded = Dict{Tuple{String,Int},Any}()
for nthreads in THREADS, impl in IMPLS
    f = joinpath(OUTDIR, "$(impl.name)_t$(nthreads).json")
    isfile(f) || continue
    d = JSON.parsefile(f)
    loaded[(impl.name, nthreads)] = (meta = d["meta"], res = Dict(e["key"] => e for e in d["results"]))
end
isempty(loaded) && error("no results in $OUTDIR")
names = [i.name for i in IMPLS]
base = first(names)

open(TABLE, "w") do io
    println(io, "# FFT implementations side by side\n")
    println(io, "Rendered $(now(UTC)) UTC on $(Sys.CPU_NAME) ($(Sys.ARCH)), Julia $(VERSION).\n")
    println(io, "| column | implementation |")
    println(io, "|:--|:--|")
    for impl in IMPLS
        ms = [loaded[k].meta for k in keys(loaded) if k[1] == impl.name]
        isempty(ms) && continue
        println(io, "| `$(impl.name)` | $(first(ms)["version"]) — `$(impl.spec)` |")
    end
    println(io, "\nTimes are minimum planned-execution times (`mul!`, or `*` where a plan has no `mul!`). ",
                "`/$(base)` is the time divided by the `$(base)` time (lower is better); ",
                "`vs $(REF)` is the speedup over the `$(REF)` column (higher is better). ",
                "✗ = threw. A checksum mismatch against `$(base)` is flagged with ⚠.\n")
    for nthreads in THREADS
        cols = [n for n in names if haskey(loaded, (n, nthreads))]
        isempty(cols) && continue
        length(THREADS) > 1 && println(io, "\n## $nthreads thread$(nthreads > 1 ? "s" : "")\n")
        keyset = String[]
        for n in cols, k in keys(loaded[(n, nthreads)].res)
            k in keyset || push!(keyset, k)
        end
        # keep the worker's order: sort by the first column that has the key
        order(k) = begin
            for n in cols
                r = loaded[(n, nthreads)].res
                haskey(r, k) && return (r[k]["shape"], r[k]["kind"], r[k]["T"], r[k]["measure"], r[k]["n"], k)
            end
        end
        sort!(keyset; by = order)
        hdr = "| case | " * join(("`$n`" for n in cols), " | ")
        hdr *= " | " * join(("`$n`/`$base`" for n in cols if n != base), " | ")
        hdr *= " | " * join(("`$n` vs `$REF`" for n in cols if n != REF && n != base), " | ") * " |"
        println(io, hdr)
        println(io, "|:--|" * "--:|"^(count(c -> true, cols) + count(!=(base), cols) + count(n -> n != REF && n != base, cols)))
        stats = Dict{Any,Vector{Float64}}()     # (group, col, :base|:ref) => ratios
        for k in keyset
            es = Dict(n => get(loaded[(n, nthreads)].res, k, nothing) for n in cols)
            t(n) = (e = es[n]; e === nothing || !haskey(e, "exec_min")) ? nothing : e["exec_min"]
            cell(n) = begin
                e = es[n]
                e === nothing && return "—"
                haskey(e, "exec_min") || return "✗"
                s = fmt_t(e["exec_min"])
                eb = es[base]
                if n != base && eb !== nothing && haskey(eb, "norm") && haskey(e, "norm")
                    tol = (e["T"] == "Float32" ? 1e-3 : 1e-8) * eb["norm"] * sqrt(e["n"])
                    (abs(e["norm"] - eb["norm"]) > tol || abs(e["sum_re"] - eb["sum_re"]) > tol ||
                     abs(e["sum_im"] - eb["sum_im"]) > tol) && (s *= " ⚠")
                end
                s
            end
            row = "| $k | " * join((cell(n) for n in cols), " | ")
            e1 = first(e for e in values(es) if e !== nothing)
            grp = (label(e1), e1["shape"] == "1d" ? e1["class"] : e1["shape"], e1["measure"])
            for n in cols
                n == base && continue
                tb, tn = t(base), t(n)
                if tb !== nothing && tn !== nothing
                    row *= " | " * fmt_r(tn / tb)
                    push!(get!(stats, (grp, n, :base), Float64[]), tn / tb)
                else
                    row *= " | —"
                end
            end
            for n in cols
                (n == REF || n == base) && continue
                tr, tn = t(REF), t(n)
                if tr !== nothing && tn !== nothing
                    sp = tr / tn
                    row *= " | " * (abs(sp - 1) > 0.05 ? "**" * fmt_r(sp) * "**" : fmt_r(sp))
                    push!(get!(stats, (grp, n, :ref), Float64[]), sp)
                else
                    row *= " | —"
                end
            end
            println(io, row * " |")
        end
        # summary
        println(io, "\n### Summary ($nthreads thread$(nthreads > 1 ? "s" : ""))\n")
        println(io, "Geometric means over the cases of each group; `(min–max)` in brackets.\n")
        shdr = "| group | " * join(("`$n`/`$base`" for n in cols if n != base), " | ")
        shdr *= " | " * join(("`$n` vs `$REF`" for n in cols if n != REF && n != base), " | ") * " |"
        println(io, shdr)
        println(io, "|:--|" * "--:|"^(count(!=(base), cols) + count(n -> n != REF && n != base, cols)))
        groups = unique(k[1] for k in keys(stats))
        sort!(groups; by = g -> (g[3], g[1], g[2]))
        for g in groups
            row = "| $(g[1]) $(g[2])$(g[3] ? " (FFTW MEASURE)" : "") |"
            for n in cols
                n == base && continue
                v = get(stats, (g, n, :base), Float64[])
                row *= isempty(v) ? " — |" : @sprintf(" %.2f× (%.2f–%.2f) |", gmean(v), minimum(v), maximum(v))
            end
            for n in cols
                (n == REF || n == base) && continue
                v = get(stats, (g, n, :ref), Float64[])
                row *= isempty(v) ? " — |" : @sprintf(" **%.2f×** (%.2f–%.2f) |", gmean(v), minimum(v), maximum(v))
            end
            println(io, row)
        end
        row = "| **all** |"
        for n in cols
            n == base && continue
            v = vcat((v for (k, v) in stats if k[2] == n && k[3] == :base)...)
            row *= isempty(v) ? " — |" : @sprintf(" %.2f× (%.2f–%.2f) |", gmean(v), minimum(v), maximum(v))
        end
        for n in cols
            (n == REF || n == base) && continue
            v = vcat((v for (k, v) in stats if k[2] == n && k[3] == :ref)...)
            row *= isempty(v) ? " — |" : @sprintf(" **%.2f×** (%.2f–%.2f), %d cases, %d slower by >5%% |", gmean(v), minimum(v), maximum(v), length(v), count(<(0.95), v))
        end
        println(io, row)
    end
    # thread scaling
    if length(THREADS) > 1
        println(io, "\n## Thread scaling (time at 1 thread / time at N threads)\n")
        println(io, "| case | " * join(("`$n` ×$t" for n in names for t in THREADS if t > 1 && haskey(loaded, (n, t))), " | ") * " |")
        println(io, "|:--|" * "--:|"^count(t -> t > 1, [t for n in names for t in THREADS if haskey(loaded, (n, t))]))
        keys1 = String[]
        for n in names, t in THREADS
            t > 1 && haskey(loaded, (n, t)) || continue
            for k in keys(loaded[(n, t)].res)
                k1 = replace(k, r" \(\d+ thr\)" => "")
                k1 in keys1 || push!(keys1, k1)
            end
        end
        for k1 in keys1
            row = "| $k1 |"
            any = false
            for n in names, t in THREADS
                t > 1 && haskey(loaded, (n, t)) || continue
                r1 = haskey(loaded, (n, 1)) ? get(loaded[(n, 1)].res, k1, nothing) : nothing
                rt = get(loaded[(n, t)].res, k1 * " ($t thr)", nothing)
                if r1 !== nothing && rt !== nothing && haskey(r1, "exec_min") && haskey(rt, "exec_min")
                    row *= " " * fmt_r(r1["exec_min"] / rt["exec_min"]) * " |"; any = true
                else
                    row *= " — |"
                end
            end
            any && println(io, row)
        end
    end
end
println("wrote $TABLE")
print(read(TABLE, String))
