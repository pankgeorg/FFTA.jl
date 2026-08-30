# Large power-of-two transforms take the leaves-first path (src/leaffirst.jl).
# Checked without FFTW (loading it here would capture `plan_*` dispatch for the
# rest of the suite): a length-n transform is rebuilt from the two half-length
# transforms of its even and odd elements (which, for the smallest n here, take
# the plain recursion), real transforms against the complex one, and the
# backward transform against the identity.
using FFTA, LinearAlgebra, Test

function rebuilt_fft(x::AbstractVector{Complex{T}}) where {T}
    n = length(x)
    E = fft(x[1:2:end]); O = fft(x[2:2:end])
    m = n ÷ 2
    X = similar(x)
    for k in 0:n-1
        w = Complex{T}(cispi(-2 * T(k) / T(n)))
        X[k + 1] = E[k % m + 1] + w * O[k % m + 1]
    end
    return X
end

@testset "leaves-first order, n = 2^$k, $T" for k in 18:21, T in (Float64, Float32)
    n = 1 << k
    rtol = T === Float64 ? 1e-9 : 1e-3
    x = randn(Complex{T}, n)
    p = plan_fft(x)
    y = p * x
    @test y ≈ rebuilt_fft(x) rtol = rtol
    @test bfft(y) ≈ n .* x rtol = rtol
    @test (@allocated mul!(y, p, x)) == 0
    xr = randn(T, n)
    pr = plan_rfft(xr)
    yr = pr * xr
    @test yr ≈ fft(complex(xr))[1:n÷2+1] rtol = rtol
    @test brfft(yr, n) ≈ n .* xr rtol = rtol
    @test (@allocated mul!(yr, pr, xr)) == 0
end
