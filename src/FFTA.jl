module FFTA

using AbstractFFTs: AbstractFFTs
using DocStringExtensions: TYPEDEF, TYPEDSIGNATURES
using LinearAlgebra: LinearAlgebra
using MuladdMacro: @muladd
using Polyester: @batch
using Primes: Primes
using Reexport: @reexport
using SIMD: Vec, vload, vstore, shufflevector

@reexport using AbstractFFTs

include("callgraph.jl")
include("singleton_twiddle.jl")
include("codelets.jl")
include("odd_codelets.jl")
include("simd_pass.jl")
include("leaffirst.jl")
include("algos.jl")
include("real_simd.jl")
include("plan.jl")

# Compile the codelets (and the common plan/execute paths) at precompile time
# rather than on first use: each (size, element type, direction) codelet costs
# 0.1-0.6 s of LLVM time.
using PrecompileTools: @setup_workload, @compile_workload
@setup_workload begin
    @compile_workload begin
        for T in (Float64, Float32)
            for n in (8, 16, 32, 64, 128, 256, 5, 7, 11, 13, 25, 49)
                x = ones(Complex{T}, n)
                y = AbstractFFTs.fft(x)
                AbstractFFTs.bfft(y)
                xr = ones(T, n)
                AbstractFFTs.irfft(AbstractFFTs.rfft(xr), n)
            end
        end
    end
end
end
