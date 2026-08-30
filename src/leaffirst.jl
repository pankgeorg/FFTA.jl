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


# ---------------------------------------------------------------------------
# Threaded leaves-first order for a single large transform
#
# The two stages above are embarrassingly parallel: the sub-transforms are
# independent (each chunk of groups uses its own gather buffer, one per
# worker), and a butterfly pass is independent across blocks and, within a
# block, across butterflies. Passes over few large blocks are split by
# butterfly range in multiples of 64 (SIMD-friendly), so the operations and
# their order per element are the same as in the serial code: results do not
# depend on the number of threads.
# ---------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)
`_pow2_leaffirst!` on `nt = length(gathers)` threads (Polyester), `gathers`
being one gather buffer per worker (see `Worker.gathers`).
"""
function _pow2_leaffirst_threaded!(
    out::AbstractVector{T}, in::AbstractVector{U},
    N::Int, start_out::Int, start_in::Int, stride_in::Int,
    d::Direction, tw::AbstractVector{T}, toff::Int, gathers::Vector{Vector{T}}
) where {T<:Complex, U}
    nt = length(gathers)
    G = _leaffirst_group(T)
    B = N
    toffB = toff
    while B > LEAFFIRST_BLOCK
        toffB += 3 * (B ÷ 4)
        B ÷= 4
    end
    P = N ÷ B
    _lf_subtransforms!(out, in, B, P, G, start_out, start_in, stride_in, d, tw, toffB, gathers)
    M = 4B
    while M <= N
        toffM = toff
        L = N
        while L > M
            toffM += 3 * (L ÷ 4)
            L ÷= 4
        end
        _lf_level!(out, N, M, start_out, d, tw, toffM, nt)
        M *= 4
    end
    return nothing
end

# (chunks are large here — a whole transform of ≥ 2^18 elements split nt
# ways — so plain tasks are used; Polyester's `@batch` cannot pass a vector
# of buffers to its threads on every platform)

# stage 1: the P sub-transforms of size B in groups of G, one chunk of groups per thread
function _lf_subtransforms!(out::AbstractVector{T}, in::AbstractVector, B::Int, P::Int, G::Int,
                            start_out::Int, start_in::Int, stride_in::Int, d::Direction,
                            tw::AbstractVector{T}, toffB::Int, gathers::Vector{Vector{T}}) where {T}
    nt = length(gathers)
    digits = trailing_zeros(P) ÷ 2
    ngroups = P ÷ G
    Base.@sync for c in 1:nt
        buf = gathers[c]
        g0 = (c - 1) * ngroups ÷ nt
        g1 = c * ngroups ÷ nt - 1
        Threads.@spawn for g in g0:g1
            q0 = g * G
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
    end
    return nothing
end

# stage 2: all passes of the level whose blocks have size M
function _lf_level!(out::AbstractVector{T}, N::Int, M::Int, start_out::Int, d::Direction,
                    tw::AbstractVector{T}, toffM::Int, nt::Int) where {T}
    m = M ÷ 4
    nblocks = N ÷ M
    if nblocks >= nt
        # whole blocks per thread
        Base.@sync for c in 1:nt
            b0 = (c - 1) * nblocks ÷ nt
            b1 = c * nblocks ÷ nt - 1
            Threads.@spawn for b in b0:b1
                _pow2_pass!(out, m, start_out + b * M, 1, d, tw, toffM)
            end
        end
    else
        # few blocks: split the butterflies of every block, in multiples of 64
        Base.@sync for c in 1:nt
            k0 = ((c - 1) * m ÷ nt) ÷ 64 * 64
            k1 = c == nt ? m : (c * m ÷ nt) ÷ 64 * 64
            k1 > k0 || continue
            Threads.@spawn for b in 0:nblocks-1
                _pow2_pass!(out, m, start_out + b * M, 1, d, tw, toffM, k0, k1)
            end
        end
    end
    return nothing
end

# Whether the root transform of `g` (size `N`) can take the threaded
# leaves-first path with the given gather buffers.
_threaded_1d_ok(g::CallGraph, gathers::Vector) =
    length(gathers) > 1 && g[1].type === POW2RADIX4_FFT && g[1].sz >= LEAFFIRST_MIN &&
    !isempty(gathers[1]) && g[1].sz ÷ LEAFFIRST_BLOCK >= 4 * length(gathers) ÷ 4 + 1
