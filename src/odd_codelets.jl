# Straight-line codelets for the small odd prime leaves (5, 7, 11, 13).
#
# The generic `fft_dft!` leaf is an O(N²) loop over a twiddle table: for a
# 5-point transform that is 16 complex multiplications from memory-loaded
# twiddles. These codelets use the symmetry of the odd-length DFT instead:
# with `a_j = x_j + x_{N-j}` and `b_j = x_j - x_{N-j}` for `j = 1..(N-1)/2`,
#
#     X_k     = x_0 + Σ_j a_j cos(2πjk/N) + D·i Σ_j b_j sin(2πjk/N)
#     X_{N-k} = x_0 + Σ_j a_j cos(2πjk/N) - D·i Σ_j b_j sin(2πjk/N)
#
# so each output pair costs `(N-1)/2` real-by-complex products for the cosine
# sum and as many for the sine sum: `(N-1)²` real multiplications per
# transform instead of `4(N-1)²`, with every constant folded and no table
# loads. Like the power-of-two codelets they are compiled once per
# `(N, element type, direction)` (see the `PrecompileTools` workload) and are
# only used for `Float32`/`Float64` elements; other types keep `fft_dft!`.
# The input may be real (the odd-length real transform runs the complex
# kernel on real input), in which case the sums are real and the same
# expressions are emitted.

const ODD_CODELET_SIZES = (5, 7, 11, 13)

# real coefficient × (real or complex) value + (real or complex) accumulator
@inline _fma(a::T, b::T, c::T) where {T<:Real} = fma(a, b, c)
@inline _fma(a::T, b::Complex{T}, c::Complex{T}) where {T<:Real} = Complex(fma(a, real(b), real(c)), fma(a, imag(b), imag(c)))
@inline _fma(a::T, b::T, c::Complex{T}) where {T<:Real} = Complex(fma(a, b, real(c)), imag(c))
@inline _fma(a::T, b::Complex{T}, c::T) where {T<:Real} = Complex(fma(a, real(b), c), a * imag(b))

# Emit the statements of the length-`N` DFT of the symbols `xs` in direction
# `dir` (`-1` forward), returning the output symbols.
function _gen_odd!(stmts::Vector{Any}, xs::Vector{Symbol}, ::Type{T}, dir::Int, counter::Ref{Int}) where {T}
    N = length(xs)
    h = (N - 1) ÷ 2
    newsym() = Symbol(:t, counter[] += 1)
    as = Vector{Symbol}(undef, h); bs = Vector{Symbol}(undef, h)
    for j in 1:h
        as[j] = newsym(); bs[j] = newsym()
        push!(stmts, :($(as[j]) = $(xs[j + 1]) + $(xs[N - j + 1])), :($(bs[j]) = $(xs[j + 1]) - $(xs[N - j + 1])))
    end
    outs = Vector{Symbol}(undef, N)
    # X_0
    acc = xs[1]
    for j in 1:h
        t = newsym(); push!(stmts, :($t = $acc + $(as[j]))); acc = t
    end
    outs[1] = acc
    for k in 1:h
        # cosine sum (real coefficients), starting from x_0; explicit fma (not
        # muladd) so that rounding does not depend on whether LLVM contracts
        # for a particular array type
        c = xs[1]
        for j in 1:h
            coef = T(cospi(2 * (j * k % N) / N))
            t = newsym(); push!(stmts, :($t = _fma($coef, $(as[j]), $c))); c = t
        end
        # sine sum, with the direction folded into the coefficients
        s = nothing
        for j in 1:h
            coef = T(dir * sinpi(2 * (j * k % N) / N))
            t = newsym()
            push!(stmts, s === nothing ? :($t = $coef * $(bs[j])) : :($t = _fma($coef, $(bs[j]), $s)))
            s = t
        end
        # ± i·s
        is = newsym(); push!(stmts, :($is = Complex(-imag($s), real($s))))
        o1 = newsym(); o2 = newsym()
        push!(stmts, :($o1 = $c + $is), :($o2 = $c - $is))
        outs[k + 1] = o1; outs[N - k + 1] = o2
    end
    return outs
end

"""
$(TYPEDSIGNATURES)
Straight-line length-`N` DFT (odd `N`) of `in[start_in + k*stride_in]` into
`out[start_out + k*stride_out]`, `k = 0..N-1`, in direction `D` (`-1`
forward, `+1` backward). The input may be real or complex.
"""
@generated function fft_odd_codelet!(
    out::AbstractVector{Complex{T}}, in::AbstractVector{<:Union{T,Complex{T}}},
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
    outs = _gen_odd!(stmts, xs, T, D, counter)
    for i in 1:N
        push!(stmts, :(out[start_out + $(i - 1) * stride_out] = $(outs[i])))
    end
    body = Expr(:block, stmts...)
    return quote
        @inbounds $body
        return nothing
    end
end

# Dispatch an odd leaf to its codelet; `false` when there is none for this
# `N` or element type (the caller then uses `fft_dft!`).
@inline function _odd_codelet!(
    out::AbstractVector{Complex{S}}, in::AbstractVector{<:Union{S,Complex{S}}}, N::Int,
    start_out::Int, stride_out::Int, start_in::Int, stride_in::Int, d::Direction
) where {S<:Union{Float32,Float64}}
    if d === FFT_FORWARD
        N == 5  && (fft_odd_codelet!(out, in, Val(5),  start_out, stride_out, start_in, stride_in, Val(-1)); return true)
        N == 7  && (fft_odd_codelet!(out, in, Val(7),  start_out, stride_out, start_in, stride_in, Val(-1)); return true)
        N == 11 && (fft_odd_codelet!(out, in, Val(11), start_out, stride_out, start_in, stride_in, Val(-1)); return true)
        N == 13 && (fft_odd_codelet!(out, in, Val(13), start_out, stride_out, start_in, stride_in, Val(-1)); return true)
    else
        N == 5  && (fft_odd_codelet!(out, in, Val(5),  start_out, stride_out, start_in, stride_in, Val(1)); return true)
        N == 7  && (fft_odd_codelet!(out, in, Val(7),  start_out, stride_out, start_in, stride_in, Val(1)); return true)
        N == 11 && (fft_odd_codelet!(out, in, Val(11), start_out, stride_out, start_in, stride_in, Val(1)); return true)
        N == 13 && (fft_odd_codelet!(out, in, Val(13), start_out, stride_out, start_in, stride_in, Val(1)); return true)
    end
    return false
end
_odd_codelet!(out, in, N, start_out, stride_out, start_in, stride_in, d) = false
