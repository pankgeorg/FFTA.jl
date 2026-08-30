#!/usr/bin/env julia
#=
Where the time goes in an N-d transform: one pass along each dimension,
FFTA vs FFTW, for the shapes DSP.jl's convolutions produce.

    julia --project=. -t 1 nd_stages.jl [--shapes 140x140,140x140x140,256x256]

Loads FFTA and FFTW in one process (FFTA reached with `invoke`, as in
suite.jl) and, for each shape, times the transform along dimension 1 only,
along dimension 2 only, …, and the full N-d transform, complex and real;
next to each: FFTW's time for the same pass and the time of a single 1-D
transform of that length times the number of pencils (what the pass would
cost with no pencil overhead at all).
=#
import Pkg
Pkg.develop(path = joinpath(@__DIR__, "..")); Pkg.instantiate()
using FFTA, FFTW, LinearAlgebra, Printf

function getopt(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
const SHAPES = [Tuple(parse.(Int, split(s, "x"))) for s in split(getopt("--shapes", "140x140,256x256,140x140x140,64x64x64"), ",")]
function timeit(f; seconds = 0.3)
    f(); f()
    best = Inf; dl = time_ns() + seconds * 1e9; n = 0
    while time_ns() < dl || n < 3
        t = @elapsed f(); best = min(best, t); n += 1
    end
    best
end
fa(x, dims) = invoke(AbstractFFTs.plan_fft, Tuple{AbstractArray{eltype(x),ndims(x)}, Any}, x, dims; num_threads = 1)
ra(x, dims) = invoke(AbstractFFTs.plan_rfft, Tuple{AbstractArray{eltype(x),ndims(x)}, Any}, x, dims; num_threads = 1)
fw(x, dims) = invoke(AbstractFFTs.plan_fft, Tuple{StridedArray{eltype(x),ndims(x)}, Any}, x, dims)
rw(x, dims) = invoke(AbstractFFTs.plan_rfft, Tuple{StridedArray{eltype(x),ndims(x)}, Any}, x, dims)
fmt(t) = t < 1e-3 ? @sprintf("%.1f µs", t * 1e6) : @sprintf("%.2f ms", t * 1e3)
exec(p, x) = (y = p * x; timeit(() -> mul!(y, p, x)))

println("| shape | pass | FFTW | FFTA | FFTA / FFTW | FFTA 1-D × pencils | pencil overhead |")
println("|:--|:--|--:|--:|--:|--:|--:|")
for sz in SHAPES, T in (ComplexF64, Float64)
    x = randn(T, sz)
    N = length(sz)
    for d in 1:N
        n = sz[d]; npencils = prod(sz) ÷ n
        if T <: Complex
            tw = exec(fw(x, d), x); ta = exec(fa(x, d), x)
            v = randn(T, n); t1 = exec(fa(v, 1), v) * npencils
        else
            d == 1 || continue                          # real transforms are along dim 1 (then complex along the rest)
            tw = exec(rw(x, d), x); ta = exec(ra(x, d), x)
            v = randn(T, n); t1 = exec(ra(v, 1), v) * npencils
        end
        @printf("| %s %s | dim %d (%d pencils of %d) | %s | **%s** | %.2f× | %s | %.2f× |\n", T, join(sz, "×"), d, npencils, n, fmt(tw), fmt(ta), ta / tw, fmt(t1), ta / t1)
    end
    tw = T <: Complex ? exec(fw(x, 1:N), x) : exec(rw(x, 1:N), x)
    ta = T <: Complex ? exec(fa(x, 1:N), x) : exec(ra(x, 1:N), x)
    @printf("| %s %s | **all dims** | %s | **%s** | %.2f× | | |\n", T, join(sz, "×"), fmt(tw), fmt(ta), ta / tw)
    flush(stdout)
end
