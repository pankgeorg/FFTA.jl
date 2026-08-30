# Leaves-first order for large power-of-two transforms.
#
# The depth-first radix-4 recursion of `fft_pow2_radix4!` reads each leaf's
# `CODELET_MAX` inputs at stride `N ÷ CODELET_MAX`: consecutive leaves in
# recursion order are a quarter of the array apart in the input, so every
# cache line fetched for a leaf is used for one element and evicted before
# the leaves that need its neighbours run. Once the array is out of the last
# cache level that costs the leaves 2–2.5× their in-cache time (measured at
# 2^20–2^22 elements, ComplexF64) and they become the largest stage of the
# transform.
#
# Above `LEAFFIRST_MIN` the transform is therefore computed as `P = N ÷ B`
# sub-transforms of size `B` (`LEAFFIRST_BLOCK` or half of it, a level of the
# recursion; decimated inputs at stride `P`), taken in input
# order in groups of `G` — one cache line of consecutive inputs — which are
# gathered into a contiguous buffer and transformed in cache, followed by the
# `log4(P)` remaining butterfly passes over the whole array. The output is
# identical to the recursion's (same operations, same order per element).
# Measured on a Neoverse-N1 (ComplexF64, with the SIMD butterfly pass):
# 2^20 43 → 26 ms, 2^22 188 → 124 ms.

const LEAFFIRST_MIN = 1 << 18    # transforms with fewer elements keep the recursion
const LEAFFIRST_BLOCK = 1 << 12  # size of the contiguous sub-transforms

# pencils gathered together: one 64-byte cache line of consecutive inputs
_leaffirst_group(::Type{T}) where {T} = max(4, 64 ÷ sizeof(T))

"""
$(TYPEDSIGNATURES)
Length of the gather buffer a `POW2RADIX4_FFT` node of size `N` keeps in its
workspace: `0` below `LEAFFIRST_MIN`.
"""
leaffirst_buflen(::Type{T}, N::Int) where {T} =
    N >= LEAFFIRST_MIN ? _leaffirst_group(T) * LEAFFIRST_BLOCK : 0

# base-4 digit reversal of `q` over `digits` digits
@inline function _rev4(q::Int, digits::Int)
    r = 0
    for _ in 1:digits
        r = 4r + (q & 3)
        q >>= 2
    end
    return r
end

function _pow2_leaffirst!(
    out::AbstractVector{T}, in::AbstractVector{U},
    N::Int, start_out::Int, start_in::Int, stride_in::Int,
    d::Direction, tw::AbstractVector{T}, toff::Int, buf::AbstractVector{T}
) where {T<:Complex, U}
    # the sub-transform size is the recursion's own block size at the level
    # nearest LEAFFIRST_BLOCK (LEAFFIRST_BLOCK or half of it, depending on the
    # parity of log2 N), so that P = N ÷ B is a power of 4
    G = _leaffirst_group(T)
    B = N
    toffB = toff
    while B > LEAFFIRST_BLOCK
        toffB += 3 * (B ÷ 4)
        B ÷= 4
    end
    P = N ÷ B
    digits = trailing_zeros(P) ÷ 2
    # (P ≥ G is guaranteed by LEAFFIRST_MIN ≥ 4·G·LEAFFIRST_BLOCK)
    # 1. sub-transforms of size B, in input order, G at a time through `buf`
    for q0 in 0:G:P-1
        @inbounds for j in 0:B-1
            src = start_in + (q0 + j * P) * stride_in
            for r in 0:G-1
                buf[r * B + j + 1] = in[src + r * stride_in]
            end
        end
        for r in 0:G-1
            fft_pow2_radix4!(out, buf, B, start_out + B * _rev4(q0 + r, digits), 1, 1 + r * B, 1, d, tw, toffB)
        end
    end
    # 2. the remaining butterfly passes, one level at a time (`_pow2_level!`
    #    descends from the top level's table offset)
    M = 4B
    while M <= N
        _pow2_level!(out, N, M, start_out, d, tw, toff)
        M *= 4
    end
    return nothing
end

# all radix-4 passes of the level whose blocks have size `L`, inside the
# block of size `N` at `start_out` (unit stride)
function _pow2_level!(out::AbstractVector{T}, N::Int, L::Int, start_out::Int, d::Direction,
                      tw::AbstractVector{T}, toff::Int) where {T}
    if N == L
        _pow2_pass!(out, N ÷ 4, start_out, 1, d, tw, toff)
        return
    end
    m = N ÷ 4
    for q in 0:3
        _pow2_level!(out, m, L, start_out + q * m, d, tw, toff + 3m)
    end
end
