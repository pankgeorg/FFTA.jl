# Vectorised pre/post-processing of the even-length real transforms.
#
# `_rfft_pencil!` turns the half-length complex transform `Y` into the real
# transform with, for `j = 2..m/2+1`,
#
#     XX = (Y[j] + conj(Y[m-j+2]))/2,  XY = i (conj(Y[m-j+2]) - Y[j])/2
#     X[j] = XX + w^j XY,  X[m-j+2] = conj(XX - w^j XY)
#
# and `_brfft_pencil!` the inverse. On scalar `Complex` values these loops are
# compute bound; here `W` values of `j` are handled per iteration on vectors
# of `2W` reals (`W = 2` for `Float64`, `4` for `Float32`): the front block is
# loaded in order, the mirrored back block is loaded and reversed lane-wise,
# conjugation is a sign flip of the imaginary lanes and `i·z` a lane swap with
# a sign flip. A scalar tail handles the remainder and the middle element.
# Used for unit-stride vectors of `ComplexF32`/`ComplexF64`; other cases keep
# the scalar loops.

# `x[2j-1], x[2j]` pairs of a unit-stride real vector are the memory layout of
# the complex vector `buf`: pack and unpack with a copy where possible.
function _pack_pairs!(buf::AbstractVector{Complex{R}}, x::AbstractVector{R}, m::Int) where {R<:Real}
    if _unit_stride(x) && _unit_stride(buf)
        GC.@preserve buf x unsafe_copyto!(Ptr{R}(pointer(buf)), pointer(x), 2m)
    else
        @inbounds for j in 1:m
            buf[j] = Complex{R}(x[2j - 1], x[2j])
        end
    end
    return buf
end
function _unpack_pairs!(x::AbstractVector{R}, out::AbstractVector{Complex{R}}, m::Int) where {R<:Real}
    if _unit_stride(x) && _unit_stride(out)
        GC.@preserve x out unsafe_copyto!(pointer(x), Ptr{R}(pointer(out)), 2m)
    else
        @inbounds for j in 1:m
            x[2j - 1] = real(out[j])
            x[2j]     = imag(out[j])
        end
    end
    return x
end
_pack_pairs!(buf, x, m) = (@inbounds for j in 1:m; buf[j] = eltype(buf)(x[2j - 1], x[2j]); end; buf)
_unpack_pairs!(x, out, m) = (@inbounds for j in 1:m; x[2j - 1] = real(out[j]); x[2j] = imag(out[j]); end; x)

# unit-stride dense storage that `pointer` can address (also used by the
# pencil loops of plan.jl)
_unit_stride(v::StridedArray) = stride(v, 1) == 1
_unit_stride(v) = false

@inline _swap_ri(v::Vec{L}) where {L} = shufflevector(v, Val(ntuple(i -> isodd(i) ? i : i - 2, L)))
@inline _rev_complex(v::Vec{L}) where {L} = shufflevector(v, Val(ntuple(i -> isodd(i) ? L - i - 1 : L - i + 1, L)))
@inline _dup_re(v::Vec{L}) where {L} = shufflevector(v, Val(ntuple(i -> 2 * ((i - 1) ÷ 2), L)))
@inline _dup_im(v::Vec{L}) where {L} = shufflevector(v, Val(ntuple(i -> 2 * ((i - 1) ÷ 2) + 1, L)))

"""
$(TYPEDSIGNATURES)
The post-processing loop of `_rfft_pencil!` for `j = 2..m÷2+1` in place on
`y`, with the twiddles `rtw[j-1] = w^(j-1)`. Returns `false` (nothing done)
when the arrays are not dense vectors of a SIMD element type.
"""
function _rfft_post_simd!(y::AbstractVector{T}, m::Int, rtw::AbstractVector{T}) where {T<:CodeletEltype}
    (_unit_stride(y) && rtw isa Vector{T}) || return false
    R = real(T)
    W = _simd_width(T)
    L = 2W
    V = Vec{L,R}
    sz = sizeof(R)
    conjs = Vec{L,R}(ntuple(i -> isodd(i) ? one(R) : -one(R), L))
    half = Vec{L,R}(R(0.5))
    py = Ptr{R}(pointer(y)); pt = Ptr{R}(pointer(rtw))
    jlast = (m >> 1) + 1
    j = 2
    GC.@preserve y rtw begin
        @inbounds while j + W - 1 < jlast
            pf = py + (j - 1) * 2sz
            pb = py + (m - j + 2 - W) * 2sz
            yj = vload(V, pf)
            ymj = _rev_complex(vload(V, pb)) * conjs      # conj(y[m-j+2]) in j order
            w = vload(V, pt + (j - 2) * 2sz)
            wr = _dup_re(w); wi = _dup_im(w) * -conjs      # (-wi, wi)
            XX = half * (yj + ymj)
            XY = _swap_ri(half * (ymj - yj)) * -conjs      # i (conj(ymj) - yj)/2
            t = muladd(_swap_ri(XY), wi, XY * wr)          # w * XY
            vstore(XX + t, pf)
            vstore(_rev_complex((XX - t) * conjs), pb)
            j += W
        end
        @inbounds for jj in j:jlast
            yj = y[jj]; ymj = y[m - jj + 2]; wj = rtw[jj - 1]
            XX = R(0.5) * ( yj + conj(ymj))
            XY = R(0.5) * (-yj + conj(ymj)) * im
            y[jj]         =      XX + wj * XY
            y[m - jj + 2] = conj(XX - wj * XY)
        end
    end
    return true
end
_rfft_post_simd!(y, m, rtw) = false

"""
$(TYPEDSIGNATURES)
The pre-processing loop of `_brfft_pencil!` for `j = 2..m÷2+1`, from the
spectrum `y` into `tmp` (a dense vector), with the conjugated twiddles.
"""
function _brfft_pre_simd!(tmp::AbstractVector{T}, y::AbstractVector{T}, m::Int, rtw::AbstractVector{T}) where {T<:CodeletEltype}
    (_unit_stride(tmp) && _unit_stride(y) && rtw isa Vector{T}) || return false
    R = real(T)
    W = _simd_width(T)
    L = 2W
    V = Vec{L,R}
    sz = sizeof(R)
    conjs = Vec{L,R}(ntuple(i -> isodd(i) ? one(R) : -one(R), L))
    py = Ptr{R}(pointer(y)); po = Ptr{R}(pointer(tmp)); pt = Ptr{R}(pointer(rtw))
    jlast = (m >> 1) + 1
    j = 2
    GC.@preserve tmp y rtw begin
        @inbounds while j + W - 1 < jlast
            pf = py + (j - 1) * 2sz
            pb = py + (m - j + 2 - W) * 2sz
            yj = vload(V, pf)
            ymj = _rev_complex(vload(V, pb)) * conjs
            w = vload(V, pt + (j - 2) * 2sz)
            wr = _dup_re(w); wi = _dup_im(w) * conjs       # conj(w): (wi, -wi)
            XX = yj + ymj
            d = yj - ymj
            XY = muladd(_swap_ri(d), wi, d * wr)           # conj(w) * (yj - conj(ymj))
            iXY = _swap_ri(XY) * -conjs                    # i·XY
            vstore(XX + iXY, po + (j - 1) * 2sz)
            vstore(_rev_complex((XX - iXY) * conjs), po + (m - j + 2 - W) * 2sz)
            j += W
        end
        @inbounds for jj in j:jlast
            wj = conj(rtw[jj - 1])
            XX =       y[jj] + conj(y[m - jj + 2])
            XY = wj * (y[jj] - conj(y[m - jj + 2]))
            tmp[jj]         =      XX + im * XY
            tmp[m - jj + 2] = conj(XX - im * XY)
        end
    end
    return true
end
_brfft_pre_simd!(tmp, y, m, rtw) = false
