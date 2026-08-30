# SIMD radix-4 butterfly pass for the power-of-two kernel.
#
# `fft_pow2_radix4!` combines the four quarter transforms of a block with one
# pass of radix-4 butterflies. Written on scalar `Complex` values, that pass
# is compute bound on the complex multiplications (LLVM does not vectorise
# across butterflies). Here `W` butterflies are processed per iteration on
# vectors of `2W` reals (`W = 2` for `Float64`, `4` for `Float32`, one or two
# NEON/SSE registers): a complex product `a·w` becomes
# `a * (wr, wr) + swap(a) * (-wi, wi)`. The twiddle table keeps its compact
# `(w^k, w^2k, w^3k)` layout (see `pow2_twiddles`); the `W` triplets an
# iteration needs are loaded as three vectors and rearranged in registers,
# which measured as fast as an expanded table in cache and faster out of it.
#
# Used when the output block is contiguous in memory (`stride_out == 1` on a
# dense vector or contiguous view) and `N ÷ 4 >= W`; otherwise the scalar loop
# in `fft_pow2_radix4!` runs. Results agree with the scalar loop to rounding.

_simd_width(::Type{ComplexF64}) = 2
_simd_width(::Type{ComplexF32}) = 4

# (ai, ar) from (ar, ai) for every complex lane
@inline _swap(v::Vec{L}) where {L} = shufflevector(v, Val(ntuple(i -> isodd(i) ? i : i - 2, L)))
# a * w with wr = (re w, re w, ...) and wi = (-im w, im w, ...)
@inline _cmul(a, wr, wi) = muladd(_swap(a), wi, a * wr)

# The three twiddle vectors of one group of `W` butterflies: the compact table
# holds `w1 w2 w3` for each `k`, i.e. `6W` reals `(r1 i1 r2 i2 r3 i3)_k` per
# group, loaded as `u, v, w`. Returns `(wr1, wi1, wr2, wi2, wr3, wi3)` with
# `wi` already carrying the `(-, +)` sign pattern.
@inline function _twiddle_vectors(u::Vec{4,Float64}, v::Vec{4,Float64}, w::Vec{4,Float64}, sign)
    # u = (r1 i1 r2 i2)  v = (r3 i3 r1' i1')  w = (r2' i2' r3' i3')
    wr1 = shufflevector(u, v, Val((0, 0, 6, 6))); wi1 = shufflevector(u, v, Val((1, 1, 7, 7))) * sign
    wr2 = shufflevector(u, w, Val((2, 2, 4, 4))); wi2 = shufflevector(u, w, Val((3, 3, 5, 5))) * sign
    wr3 = shufflevector(v, w, Val((0, 0, 6, 6))); wi3 = shufflevector(v, w, Val((1, 1, 7, 7))) * sign
    return wr1, wi1, wr2, wi2, wr3, wi3
end
@inline function _twiddle_vectors(u::Vec{8,Float32}, v::Vec{8,Float32}, w::Vec{8,Float32}, sign)
    # u = (r1 i1 r2 i2 r3 i3 r1' i1')  v = (r2' i2' r3' i3' r1'' i1'' r2'' i2'')  w = (r3'' i3'' r1''' i1''' r2''' i2''' r3''' i3''')
    w1 = shufflevector(shufflevector(u, v, Val((0, 1, 6, 7, 12, 13, 12, 13))), w, Val((0, 1, 2, 3, 4, 5, 10, 11)))
    w2 = shufflevector(shufflevector(u, v, Val((2, 3, 8, 9, 14, 15, 14, 15))), w, Val((0, 1, 2, 3, 4, 5, 12, 13)))
    w3 = shufflevector(shufflevector(u, v, Val((4, 5, 10, 11, 4, 5, 10, 11))), w, Val((0, 1, 2, 3, 8, 9, 14, 15)))
    dupr(x) = shufflevector(x, Val((0, 0, 2, 2, 4, 4, 6, 6)))
    dupi(x) = shufflevector(x, Val((1, 1, 3, 3, 5, 5, 7, 7)))
    return dupr(w1), dupi(w1) * sign, dupr(w2), dupi(w2) * sign, dupr(w3), dupi(w3) * sign
end

_simd_contiguous(out::Array) = true
_simd_contiguous(out::SubArray) = Base.iscontiguous(out)
_simd_contiguous(out) = false

"""
$(TYPEDSIGNATURES)
The radix-4 butterfly pass of `fft_pow2_radix4!` over the `4m` outputs starting
at `out[start_out]` (unit stride), `W` butterflies per iteration. Returns
`false` without touching `out` when the pass cannot be vectorised (strided or
non-contiguous output, or `m < W`), in which case the caller runs the scalar
loop.
"""
@inline function _pow2_pass_simd!(
    out::AbstractVector{T}, m::Int, start_out::Int, stride_out::Int, d::Direction,
    tw::AbstractVector{T}, toff::Int
) where {T<:CodeletEltype}
    W = _simd_width(T)
    (stride_out == 1 && m >= W && _simd_contiguous(out) && tw isa Vector{T}) || return false
    R = real(T)
    L = 2W
    V = Vec{L,R}
    sz = sizeof(R)
    # (-, +) pattern for the imaginary parts of the twiddles; the ∓i rotation
    # of the last butterfly leg uses the opposite pattern in the forward
    # direction and the same one backward
    wsign = Vec{L,R}(ntuple(i -> isodd(i) ? -one(R) : one(R), L))
    esign = d === FFT_FORWARD ? -wsign : wsign
    po = Ptr{R}(pointer(out)) + (start_out - 1) * 2sz
    pt = Ptr{R}(pointer(tw)) + toff * 2sz
    GC.@preserve out tw begin
        @inbounds for k in 0:W:m-1
            p0 = po + k * 2sz
            p1 = p0 + m * 2sz
            p2 = p0 + 2m * 2sz
            p3 = p0 + 3m * 2sz
            y0 = vload(V, p0); y1 = vload(V, p1); y2 = vload(V, p2); y3 = vload(V, p3)
            tb = pt + 3k * 2sz
            wr1, wi1, wr2, wi2, wr3, wi3 = _twiddle_vectors(vload(V, tb), vload(V, tb + L * sz), vload(V, tb + 2L * sz), wsign)
            t1 = _cmul(y1, wr1, wi1)
            t2 = _cmul(y2, wr2, wi2)
            t3 = _cmul(y3, wr3, wi3)
            a = y0 + t2; b = y0 - t2
            c = t1 + t3; e = _swap(t1 - t3) * esign
            vstore(a + c, p0); vstore(b + e, p1); vstore(a - c, p2); vstore(b - e, p3)
        end
    end
    return true
end
_pow2_pass_simd!(out, m, start_out, stride_out, d, tw, toff) = false
