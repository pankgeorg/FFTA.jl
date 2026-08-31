# Adjoint plans: ⟨p*x, y⟩ == ⟨x, p'*y⟩ (AbstractFFTs' generic adjoint over
# the AdjointStyle methods), for complex, real forward/backward and in-place
# plans, 1-D and along a dimension.
using FFTA, LinearAlgebra, Test

@testset "adjoint plans, $T" for T in (Float64, Float32)
    rtol = T === Float64 ? 1e-12 : 1e-5
    C = Complex{T}
    x = randn(C, 24); y = randn(C, 24)
    p = plan_fft(x)
    @test dot(p * x, y) ≈ dot(x, p' * y) rtol = rtol
    @test p'' === p
    X = randn(C, 8, 6); Y = randn(C, 8, 6)
    p2 = plan_fft(X)
    @test dot(p2 * X, Y) ≈ dot(X, p2' * Y) rtol = rtol
    # (adjoints of dims-subset plans need `size(p)` to be the full array size;
    # unsupported for now, like before)
    # real-transform adjoints are real-linear: the identity holds on the
    # real part of the complex inner product (as in AbstractFFTs' own tests)
    xr = randn(T, 24); yr = randn(C, 13)
    pr = plan_rfft(xr)
    @test real(dot(pr * xr, yr)) ≈ dot(xr, pr' * yr) rtol = rtol
    pb = plan_brfft(yr, 24)
    @test dot(pb * yr, xr) ≈ real(dot(yr, pb' * xr)) rtol = rtol
    pi = plan_fft!(copy(x))
    @test dot(pi * copy(x), y) ≈ dot(x, pi' * copy(y)) rtol = rtol
end

@testset "backward real N-d mul! does not allocate" begin
    X = randn(32, 32)
    P = plan_rfft(X); Y = P * X
    IP = plan_brfft(Y, 32); Z = IP * Y
    mul!(Z, IP, Y)
    # `vec(...)` reshape wrappers of the pointer-copy pencil path
    @test (@allocated mul!(Z, IP, Y)) <= 128
    X3 = randn(16, 16, 16)
    P3 = plan_rfft(X3); Y3 = P3 * X3
    IP3 = plan_brfft(Y3, 16); Z3 = IP3 * Y3
    mul!(Z3, IP3, Y3)
    @test (@allocated mul!(Z3, IP3, Y3)) <= 128
end

@testset "adjoint over a dims subset" begin
    x = randn(ComplexF64, 8, 9, 10)
    y = randn(ComplexF64, 8, 9, 10)
    for dims in ((1,), (2,), (3,), (1, 2), (2, 3), (1, 3), (1, 2, 3))
        p = plan_fft(x, dims)
        @test dot(p * x, y) ≈ dot(x, p' * y)
        pi_ = plan_fft!(copy(x), dims)
        @test dot(pi_ * copy(x), y) ≈ dot(x, pi_' * y)
    end
    xr = randn(8, 9, 10)
    for dims in ((1,), (2,), (3,), (2, 3))
        pr = plan_rfft(xr, dims)
        yc = randn(ComplexF64, size(pr * xr))
        @test real(dot(pr * xr, yc)) ≈ real(dot(xr, pr' * yc))
        pb = plan_brfft(pr * xr, size(xr, first(dims)), dims)
        yr = randn(size(xr))
        @test real(dot(pb * (pr * xr), yr)) ≈ real(dot(pr * xr, pb' * yr))
    end
end
