# Size classes and the case list shared by `suite.jl` and `compare3.jl`.
#
# A case is a NamedTuple (kind, T, sz, dims, shape, nthreads, measure); `T` is
# the *real* element type (complex cases use `Complex{T}`).

using Primes

function size_classes(maxlog2)
    nmax = 1 << maxlog2
    pow2   = [1 << k for k in 3:maxlog2]
    # 2^a 3^b 5^c 7^d, not powers of two, spread log-uniformly
    smooth_all = Int[]
    for a in 0:maxlog2, b in 0:14, c in 0:9, d in 0:8
        n = 2^a * 3^b * 5^c * 7^d
        (n <= nmax && n >= 8 && !ispow2(n)) && push!(smooth_all, n)
    end
    sort!(unique!(smooth_all))
    smooth = Int[]
    for t in exp.(range(log(12), log(nmax), length = 18))
        push!(smooth, smooth_all[argmin(abs.(log.(smooth_all) .- log(t)))])
    end
    # explicitly include some "classic" smooth sizes
    for n in (12, 60, 120, 360, 720, 1000, 1000_000)
        n <= nmax && push!(smooth, n)
    end
    sort!(unique!(smooth))
    # primes: below (DFT path) and above (Bluestein path) the cutoff
    prime = Int[]
    for t in exp.(range(log(7), log(nmax), length = 16))
        push!(prime, nextprime(round(Int, t)))
    end
    # straddle the DFT/Bluestein crossover: the O(n²) leaf is used below the
    # cutoff (73 originally, 29 since the twiddle-table work), so sample the
    # band 23–46 as well as 61–79
    for n in (23, 29, 31, 37, 43, 61, 71, 73, 79)
        push!(prime, n)
    end
    filter!(<=(nmax), prime); sort!(unique!(prime))
    # awkward: (prime > cutoff) × small factor
    awk = Int[]
    for p in (101, 1009, 4099, 16411, 65537, 262147), f in (2, 3, 4, 6, 16)
        n = p * f
        n <= nmax && push!(awk, n)
    end
    sort!(unique!(awk))
    return (pow2 = pow2, smooth = smooth, prime = prime, awkward = awk)
end

function class_of(classes, n)
    for (k, v) in pairs(classes)
        n in v && return String(k)
    end
    return "other"
end

"""
    case_list(; only, kinds, maxlog2, sizes, nthreads, measure) -> Vector{NamedTuple}

The sweep of `suite.jl`: 1D classes × {Float64, Float32} × {fft, rfft}, a
FFTW-`MEASURE` column for 1D pow2 ComplexF64, 2D/3D, batched `dims`, and the
threaded subset. `nthreads > 1` selects the threaded subset only when `only`
contains `"threads"`; the other sections are always emitted with `nthreads`.
"""
function case_list(; only = ("1d", "nd", "batched", "threads"), kinds = (:fft, :rfft),
                     maxlog2 = 22, sizes = Int[], nthreads = 1, measure = true)
    classes = size_classes(maxlog2)
    nmax = 1 << maxlog2
    cases = NamedTuple[]
    add!(kind, T, sz, dims, shape; measure = false, thr = nthreads) =
        kind in kinds && push!(cases, (kind = kind, T = T, sz = sz, dims = dims, shape = shape,
                                        nthreads = thr, measure = measure,
                                        class = length(sz) == 1 ? class_of(classes, sz[1]) : "nd"))
    if "1d" in only
        for cls in (:pow2, :smooth, :prime, :awkward), n in classes[cls]
            isempty(sizes) || n in sizes || continue
            for T in (Float64, Float32)
                add!(:fft,  T, (n,), 1, "1d")
                add!(:rfft, T, (n,), 1, "1d")
            end
        end
        if measure
            for n in classes.pow2
                isempty(sizes) || n in sizes || continue
                add!(:fft, Float64, (n,), 1, "1d"; measure = true)
            end
        end
    end
    if "nd" in only
        nd2 = filter(n -> n * n <= nmax, [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
        for n in nd2, T in (Float64, Float32)
            add!(:fft,  T, (n, n), (1, 2), "2d")
            add!(:rfft, T, (n, n), (1, 2), "2d")
        end
        for sz in ((1000, 1000), (720, 480), (1009, 64), (64, 1009), (127, 257))
            prod(sz) <= nmax || continue
            add!(:fft,  Float64, sz, (1, 2), "2d")
            add!(:rfft, Float64, sz, (1, 2), "2d")
        end
        nd3 = filter(n -> n^3 <= nmax, [8, 16, 32, 64, 128])
        for n in nd3, T in (Float64, Float32)
            add!(:fft,  T, (n, n, n), (1, 2, 3), "3d")
            add!(:rfft, T, (n, n, n), (1, 2, 3), "3d")
        end
    end
    if "batched" in only
        for n in filter(n -> n * 64 <= nmax, [64, 256, 1024, 4096, 16384, 65536]), T in (Float64, Float32)
            add!(:fft,  T, (n, 64), 1, "batched_dim1")
            add!(:fft,  T, (64, n), 2, "batched_dim2")
            add!(:rfft, T, (n, 64), 1, "batched_dim1")
            add!(:rfft, T, (64, n), 2, "batched_dim2")
        end
        for (n, m) in ((256, 4096), (1024, 1024), (4096, 256))
            n * m <= nmax || continue
            add!(:rfft, Float64, (n, m), 1, "batched_dim1")
        end
    end
    if "threads" in only && nthreads > 1
        for n in filter(n -> n >= 1 << 16, classes.pow2)
            add!(:fft, Float64, (n,), 1, "1d")
        end
        for n in filter(n -> n * n <= nmax && n >= 256, [256, 512, 1024, 2048])
            add!(:fft, Float64, (n, n), (1, 2), "2d")
        end
        for n in filter(n -> n * 64 <= nmax && n >= 1024, [1024, 4096, 16384, 65536])
            add!(:fft, Float64, (n, 64), 1, "batched_dim1")
        end
    end
    return unique(cases)
end

casekey(c) = string(c.kind, " ", c.T, " ", join(c.sz, "×"), " dims=", join(c.dims, ","),
                    c.nthreads > 1 ? " ($(c.nthreads) thr)" : "", c.measure ? " MEASURE" : "")
