# Straight-line "codelets" for the small power-of-two base cases of
# `fft_pow2_radix4!`.
#
# A radix-2 decimation-in-time transform of size `N` is unrolled completely at
# compile time by `@generated` functions keyed on `Val{N}` and the direction:
# every intermediate value lives in its own SSA variable, twiddle factors are
# folded to constants and trivial multiplications (by 1, -1, ±i) are removed.
# This is what FFTW's `genfft` does offline; here Julia's compiler does it on
# first use for each `(N, T, direction)`, at a cost of roughly 0.1-0.6 s each
# (see the `PrecompileTools` workload at the end of the module). The codelets
# are only used for `Complex{Float32}` and `Complex{Float64}`; other element
# types keep the generic recursion, which also serves as the reference
# implementation.

const CODELET_MAX = 64
const CodeletEltype = Union{ComplexF32,ComplexF64}

# Emit the statements computing the length-`length(xs)` DFT of the symbols
# `xs` (radix-2 DIT), returning the symbols holding the outputs.
function _gen_dit!(stmts::Vector{Any}, xs::Vector{Symbol}, ::Type{T}, dir::Int, counter::Ref{Int}) where {T}
    N = length(xs)
    N == 1 && return xs
    newsym() = Symbol(:t, counter[] += 1)
    E = _gen_dit!(stmts, xs[1:2:end], T, dir, counter)
    O = _gen_dit!(stmts, xs[2:2:end], T, dir, counter)
    outs = Vector{Symbol}(undef, N)
    for k in 0:N÷2-1
        o = O[k + 1]
        t = newsym()
        if k == 0
            push!(stmts, :($t = $o))
        elseif 4k == N   # multiply by ∓i
            push!(stmts, dir < 0 ? :($t = Complex{$T}(imag($o), -real($o))) :
                                   :($t = Complex{$T}(-imag($o), real($o))))
        else
            w = cispi(dir * 2 * k / N)
            wr, wi = T(real(w)), T(imag(w))
            # explicit fma (not muladd) so that rounding does not depend on
            # whether LLVM contracts for a particular array type
            push!(stmts, :($t = Complex{$T}(fma($wr, real($o), -$wi * imag($o)),
                                           fma($wr, imag($o), $wi * real($o)))))
        end
        a, b = newsym(), newsym()
        push!(stmts, :($a = $(E[k + 1]) + $t), :($b = $(E[k + 1]) - $t))
        outs[k + 1] = a
        outs[k + 1 + N÷2] = b
    end
    return outs
end

"""
$(TYPEDSIGNATURES)
Fully unrolled length-`N` FFT of the strided input `in[start_in + k*stride_in]`
into `out[start_out + k*stride_out]`, `k = 0..N-1`, in direction `D`
(`-1` forward, `+1` backward).
"""
@generated function fft_pow2_codelet!(
    out::AbstractVector{Complex{T}}, in::AbstractVector{Complex{T}},
    ::Val{N},
    start_out::Int, stride_out::Int,
    start_in::Int, stride_in::Int,
    ::Val{D}
) where {T,N,D}
    counter = Ref(0)
    stmts = Any[]
    xs = [Symbol(:x, i) for i in 1:N]
    for i in 1:N
        push!(stmts, :($(xs[i]) = in[start_in + $(i - 1) * stride_in]))
    end
    outs = _gen_dit!(stmts, xs, T, D, counter)
    for i in 1:N
        push!(stmts, :(out[start_out + $(i - 1) * stride_out] = $(outs[i])))
    end
    body = Expr(:block, stmts...)
    return quote
        @inbounds $body
        return nothing
    end
end

# Dispatch a small power of two to its codelet. Returns `false` when there is
# no codelet for this `N` (the caller then continues with the recursion).
@inline function _pow2_codelet!(
    out::AbstractVector{T}, in::AbstractVector{T}, N::Int,
    start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction
) where {T<:CodeletEltype}
    if d === FFT_FORWARD
        N == 64 && (fft_pow2_codelet!(out, in, Val(64), start_out, stride_out, start_in, stride_in, Val(-1)); return true)
        N == 32 && (fft_pow2_codelet!(out, in, Val(32), start_out, stride_out, start_in, stride_in, Val(-1)); return true)
        N == 16 && (fft_pow2_codelet!(out, in, Val(16), start_out, stride_out, start_in, stride_in, Val(-1)); return true)
        N == 8  && (fft_pow2_codelet!(out, in, Val(8),  start_out, stride_out, start_in, stride_in, Val(-1)); return true)
    else
        N == 64 && (fft_pow2_codelet!(out, in, Val(64), start_out, stride_out, start_in, stride_in, Val(1)); return true)
        N == 32 && (fft_pow2_codelet!(out, in, Val(32), start_out, stride_out, start_in, stride_in, Val(1)); return true)
        N == 16 && (fft_pow2_codelet!(out, in, Val(16), start_out, stride_out, start_in, stride_in, Val(1)); return true)
        N == 8  && (fft_pow2_codelet!(out, in, Val(8),  start_out, stride_out, start_in, stride_in, Val(1)); return true)
    end
    return false
end
_pow2_codelet!(out, in, N, start_out, stride_out, start_in, stride_in, d) = false

# ---------------------------------------------------------------------------
# Lockstep codelets: `W` leaves at once on SIMD vectors
#
# At the level of the recursion whose four children are codelets (blocks of
# 128 or 256 points), the four sibling leaves are independent transforms of
# the same size with the same strides. They are computed together, each leaf
# in its own complex lane of a `Vec{2W}` (`W = 2` for `Float64`, `4` for
# `Float32`): the same radix-2 DIT statements as `fft_pow2_codelet!`, on
# vectors instead of scalars, with the twiddle products written so that each
# lane rounds exactly as the scalar codelet does (`muladd(wr, o, swap(o)·wi)`
# is `fma(wr, re, -wi·im)` / `fma(wr, im, wi·re)` on FMA hardware). Inputs
# are gathered lane by lane (the leaves' inputs are interleaved in memory)
# and outputs scattered; the arithmetic in between is `W`× narrower.
# Measured on Neoverse-N1 for 64 leaves of 64 points: 1.35× (`Float64`)
# and 2.4× (`Float32`) over the scalar codelets.
# ---------------------------------------------------------------------------

@inline _vswap(v::Vec{L}) where {L} = shufflevector(v, Val(ntuple(i -> isodd(i) ? i : i - 2, L)))

function _gen_dit_vec!(stmts::Vector{Any}, xs::Vector{Symbol}, ::Type{T}, L::Int, dir::Int, counter::Ref{Int}) where {T}
    N = length(xs)
    N == 1 && return xs
    newsym() = Symbol(:t, counter[] += 1)
    E = _gen_dit_vec!(stmts, xs[1:2:end], T, L, dir, counter)
    O = _gen_dit_vec!(stmts, xs[2:2:end], T, L, dir, counter)
    outs = Vector{Symbol}(undef, N)
    for k in 0:N÷2-1
        o = O[k + 1]
        t = newsym()
        if k == 0
            push!(stmts, :($t = $o))
        elseif 4k == N   # multiply by ∓i
            push!(stmts, :($t = _vswap($o) * $(dir < 0 ? :sign_pm : :sign_mp)))
        else
            w = cispi(dir * 2 * k / N)
            wr, wi = T(real(w)), T(imag(w))
            wiv = ntuple(i -> isodd(i) ? -wi : wi, L)
            push!(stmts, :($t = muladd(Vec{$L,$T}($wr), $o, _vswap($o) * Vec{$L,$T}($wiv))))
        end
        a, b = newsym(), newsym()
        push!(stmts, :($a = $(E[k + 1]) + $t), :($b = $(E[k + 1]) - $t))
        outs[k + 1] = a
        outs[k + 1 + N÷2] = b
    end
    return outs
end

"""
$(TYPEDSIGNATURES)
`W` length-`N` transforms in lockstep: leaf `q = 0..W-1` reads
`in[start_in + q*stride_in + j*4stride_in]` and writes
`out[start_out + q*N*stride_out + j*stride_out]`, `j = 0..N-1` — the strides
of four sibling leaves of the radix-4 recursion. Direction `D` as in
`fft_pow2_codelet!`.
"""
@generated function fft_pow2_codelet_lockstep!(
    out::AbstractVector{Complex{T}}, in::AbstractVector{Complex{T}},
    ::Val{N}, ::Val{W},
    start_out::Int, stride_out::Int,
    start_in::Int, stride_in::Int,
    ::Val{D}
) where {T,N,W,D}
    L = 2W
    counter = Ref(0)
    stmts = Any[]
    xs = [Symbol(:x, i) for i in 1:N]
    for i in 1:N
        parts = Any[]
        for q in 0:W-1
            c = Symbol(:c, i, :_, q)
            push!(stmts, :($c = in[start_in + $q * stride_in + $(4(i - 1)) * stride_in]))
            push!(parts, :(real($c)), :(imag($c)))
        end
        push!(stmts, :($(xs[i]) = Vec{$L,$T}($(Expr(:tuple, parts...)))))
    end
    outs = _gen_dit_vec!(stmts, xs, T, L, D, counter)
    for i in 1:N, q in 0:W-1
        push!(stmts, :(out[start_out + $(q * N) * stride_out + $(i - 1) * stride_out] = Complex{$T}($(outs[i])[$(2q + 1)], $(outs[i])[$(2q + 2)])))
    end
    body = Expr(:block, stmts...)
    return quote
        sign_pm = Vec{$L,$T}($(ntuple(i -> isodd(i) ? one(T) : -one(T), L)))
        sign_mp = -sign_pm
        @inbounds $body
        return nothing
    end
end

# The four codelet-sized children of a block of `N` points (`N` = 4 × 32 or
# 4 × 64) in lockstep; `false` when not applicable (the caller recurses).
@inline function _pow2_lockstep!(
    out::AbstractVector{T}, in::AbstractVector{T}, N::Int,
    start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction
) where {T<:CodeletEltype}
    m = N >> 2
    (m == 32 || m == 64) || return false
    vd = d === FFT_FORWARD ? Val(-1) : Val(1)
    if T === ComplexF32
        m == 64 ? fft_pow2_codelet_lockstep!(out, in, Val(64), Val(4), start_out, stride_out, start_in, stride_in, vd) :
                  fft_pow2_codelet_lockstep!(out, in, Val(32), Val(4), start_out, stride_out, start_in, stride_in, vd)
    else
        for r0 in (0, 2)
            so = start_out + r0 * m * stride_out
            si = start_in + r0 * stride_in
            m == 64 ? fft_pow2_codelet_lockstep!(out, in, Val(64), Val(2), so, stride_out, si, stride_in, vd) :
                      fft_pow2_codelet_lockstep!(out, in, Val(32), Val(2), so, stride_out, si, stride_in, vd)
        end
    end
    return true
end
_pow2_lockstep!(out, in, N, start_out, stride_out, start_in, stride_in, d) = false
