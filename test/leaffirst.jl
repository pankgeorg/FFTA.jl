# Large power-of-two transforms take the leaves-first path (src/leaffirst.jl):
# both parities of log2(n), complex and real, against FFTW.
using FFTA, FFTW, LinearAlgebra, Test

ffta_plan_fft(x) = invoke(AbstractFFTs.plan_fft, Tuple{AbstractArray{eltype(x),1}, Any}, x, 1)
ffta_plan_rfft(x) = invoke(AbstractFFTs.plan_rfft, Tuple{AbstractArray{eltype(x),1}, Any}, x, 1)

@testset "leaves-first order, n = 2^$k, $T" for k in 18:21, T in (Float64, Float32)
    n = 1 << k
    rtol = T === Float64 ? 1e-10 : 1e-4
    x = randn(Complex{T}, n)
    p = ffta_plan_fft(x)
    @test p isa FFTA.FFTAPlan
    y = p * x
    @test y ≈ FFTW.fft(x) rtol = rtol
    @test (@allocated mul!(y, p, x)) == 0
    xr = randn(T, n)
    pr = ffta_plan_rfft(xr)
    @test pr isa FFTA.FFTAPlan
    yr = pr * xr
    @test yr ≈ FFTW.rfft(xr) rtol = rtol
    @test (@allocated mul!(yr, pr, xr)) == 0
end
