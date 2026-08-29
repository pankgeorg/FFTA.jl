using FFTA, Test

@testset " forward. N=$N" for N in [8, 11, 15, 16, 27, 100]
    x = ones(Float64, N, N)
    y = rfft(x)
    y_ref = zero(y)
    y_ref[1] = length(x)
    @test y ≈ y_ref
    x = randn(N,N)
    @test rfft(x) ≈ rfft(reshape(x,1,N,N), [2,3])[1,:,:]
    @test rfft(x) ≈ rfft(reshape(x,1,N,N,1), [2,3])[1,:,:,1]
    @test rfft(x) ≈ rfft(reshape(x,1,1,N,N,1), [3,4])[1,1,:,:,1]
    @test size(rfft(x)) == (N÷2+1, N)
end

@testset "2D plan, 2D array. Size: $n" for n in 1:64
    @testset "size: ($m, $n)" for m in n:(n + 1)
        X = randn(m, n)

        @testset "against naive implementation" begin
            @test naive_2d_fourier_transform(X, FFTA.FFT_FORWARD)[1:(m ÷ 2 + 1),:] ≈ rfft(X)
        end

        @testset "allocations" begin
            @test (@test_allocations rfft(X)) <= 132 * Threads.nthreads()
        end
    end
end

@testset "2D plan, ND array. Size: $n" for n in 1:64
    x = randn(n, n + 1, n + 2)

    @testset "against 2D arrays with mapslices, r=$r" for r in [[1,2], [1,3], [2,3]]
        # `mapslices` copies each pencil to a contiguous Vector, and so does the
        # copy-in path for strided pencils, which therefore reproduces it exactly;
        # unit-stride pencils are handed to the kernels as views instead, and
        # `@muladd` contraction in the kernels can differ by an ulp between the
        # two specialisations. Any real defect (mis-strided pencil, transposed or
        # dropped copy-back) would be O(1), far outside this tolerance.
        @test rfft(x, r) ≈ mapslices(rfft, x; dims = r) rtol=1e-13
    end
end

@testset "allocations" begin
    X = randn(256, 256)
    rfft(X) # compile
    @test (@test_allocations rfft(X)) <= 63 * Threads.nthreads()
end
