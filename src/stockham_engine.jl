# Split-plane Stockham engine for smooth 1-D transforms.
#
# The classic Stockham autosort formulation (see e.g. Van Loan,
# "Computational Frameworks for the FFT"): the transform runs as a chain of
# out-of-place stages ping-ponging between two buffers, with the reordering
# permutation absorbed into each stage's output indexing, so there is no
# bit-reversal pass and every stage streams both buffers sequentially. Data
# lives in split re/im planes (`Vector{real(T)}`), which makes every stage a
# pure vertical SIMD loop — no in-register shuffles in the hot path, and
# `Float32` uses twice the lanes of `Float64` (the interleaved-complex kernels
# elsewhere in FFTA cannot do that). The interleaved complex input/output is
# converted at the edges.
#
# A stage splits the current sub-length `n` into `R` parts of `m = n ÷ R`:
# butterfly `p ∈ 0:m-1` reads lanes `q ∈ 1:s` at `x[s·p + j·m·s + q]`,
# applies the `R`-point DFT with twiddles `w = exp(2πi·D·p·j/n)` on outputs
# `j ≥ 1`, and writes `y[R·s·p + j·s + q]`. The chain is a radix-4 tower
# (with a lane-expanded layout while `s < W`), then the odd radices in
# ascending order (a generic conjugate-symmetric butterfly, the same
# formulation as `odd_codelets.jl`, serves every odd radix), then a
# twiddle-free final power-of-two stage (2, 4 or 8).
#
# Used for `Float32`/`Float64` transforms whose length factors over
# 2·3·5·7·11·13 (see `_use_stockham`); the call-graph node type is
# `STOCKHAM` and the chain lives in `CallGraph.stockham`.

const STOCKHAM_RADICES = (3, 5, 7, 11, 13)

"one radix stage: `R`, its twiddle planes, and whether they use the lane-expanded small-stride layout"
struct StockhamStage{S<:Real}
    R::Int
    small::Bool
    twr::Vector{S}
    twi::Vector{S}
end

"stage chain plus this graph's scratch planes (`_clone_workspace` gives clones fresh planes, shared stages)"
struct StockhamChain{S<:Real}
    n::Int
    stages::Vector{StockhamStage{S}}
    are::Vector{S}
    aim::Vector{S}
    bre::Vector{S}
    bim::Vector{S}
    nt::Int
end
_clone_chain(c::StockhamChain{S}) where {S} =
    StockhamChain{S}(c.n, c.stages, similar(c.are), similar(c.aim), similar(c.bre), similar(c.bim), c.nt)

# host vector register width in bits (scalable extensions such as SVE are
# not detectable here, so aarch64 assumes 128-bit NEON)
function _host_vector_bits()
    Sys.ARCH === :x86_64 || return 128
    CPUID = Base.BinaryPlatforms.CPUID
    CPUID.test_cpu_feature(CPUID.JL_X86_avx512f) && return 512
    CPUID.test_cpu_feature(CPUID.JL_X86_avx) && return 256
    return 128
end
const _STOCKHAM_VECTOR_BITS = _host_vector_bits()

_stockham_width(::Type{Float64}) = _STOCKHAM_VECTOR_BITS ÷ 64
_stockham_width(::Type{Float32}) = _STOCKHAM_VECTOR_BITS ÷ 32

# Flag-independent loads and stores: `--check-bounds=yes` (which `Pkg.test`
# uses) disables `@inbounds`, so array-form SIMD ops would pay a range check
# per vector. These go through the data pointer — the per-op `GC.@preserve`
# keeps the array rooted and compiles away.
@inline _sld(::Type{Vec{W,S}}, a, i::Int) where {W,S} =
    GC.@preserve a vload(Vec{W,S}, pointer(a) + (i - 1) * sizeof(S))
@inline _sst!(a, i::Int, v::Vec{W,S}) where {W,S} =
    GC.@preserve a vstore(v, pointer(a) + (i - 1) * sizeof(S))
@inline _sst!(a, i::Int, v) =
    GC.@preserve a unsafe_store!(pointer(a), convert(eltype(a), v), i)
@inline _sget(a, i::Int) = GC.@preserve a unsafe_load(pointer(a), i)

# Threading: each stage is one full pass over the planes, split into chunks of
# its outer index (butterfly index `p`, or W-aligned lane blocks). Chunks write
# disjoint regions and compute the same expressions, so threaded results are
# bitwise identical to serial. Only transforms at least this long thread:
const _STOCKHAM_THREAD_MIN = 1 << 17

"chunk `c` of `nt` over `1:len` (inclusive bounds; empty when `hi < lo`)"
@inline _srange(c::Int, nt::Int, len::Int) = (div((c - 1) * len, nt) + 1, div(c * len, nt))

# Chunk a (butterfly `p`, lane block) stage: split `p` when there are enough
# butterflies; otherwise split the W-aligned lane blocks (only safe when the
# stride has no overlap window, i.e. `s % W == 0`).
@inline function _spq(c::Int, nt::Int, m::Int, s::Int, W::Int)
    nb = s < W ? 1 : cld(s, W)
    if m >= 2 * nt || nb < 2 * nt
        lo, hi = _srange(c, nt, m)
        return lo, hi, 1, nb
    end
    # every chunk gets >= 2 blocks, so the final overlap window (block `nb`
    # rewrites lanes of block `nb - 1` with identical values) stays inside
    # the last chunk
    qlo, qhi = _srange(c, nt, nb)
    return 1, m, qlo, qhi
end

# Sense-reversing barrier for the per-transform task team: one spawn per
# transform, then one cheap barrier between stages instead of a fork-join
# per stage. The last arrival resets the count and bumps the generation.
mutable struct _SBarrier
    @atomic count::Int
    @atomic gen::Int
    const nt::Int
end
@inline _swait(::Nothing) = nothing
@inline function _swait(b::_SBarrier)
    g = @atomic :acquire b.gen
    if (@atomic b.count += 1) == b.nt
        @atomic b.count = 0
        @atomic :release b.gen = g + 1
    else
        spins = 0
        while (@atomic :acquire b.gen) == g
            GC.safepoint()
            ccall(:jl_cpu_pause, Cvoid, ())
            spins += 1
            if spins > 10_000   # oversubscribed scheduler: let others run
                yield()
                spins = 0
            end
        end
    end
    return nothing
end

"is `n` a product of 2 and `STOCKHAM_RADICES`?"
function _stockham_smooth(n::Int)
    r = n >> trailing_zeros(n)
    for p in STOCKHAM_RADICES
        while r % p == 0
            r ÷= p
        end
    end
    return r == 1
end

# Whether the root transform of length `N` should use this engine. Pure
# powers of two keep the radix-4/leaves-first kernel for ComplexF64 (measured
# faster) and whenever the ≥ 2^18 threaded path may run; ComplexF32 powers of
# two are faster here (twice the lanes).
function _use_stockham(::Type{T}, N::Int, num_threads::Int) where {T}
    T <: CodeletEltype || return false
    N >= 32 || return false
    _stockham_smooth(N) || return false
    if ispow2(N)
        # ComplexF32 below LEAFFIRST_MIN: this engine wins (twice the lanes).
        # ComplexF64: the radix-4 leaves-first kernel is faster serial and
        # threads at ≥ LEAFFIRST_MIN, except 2^17 where the threaded stage
        # engine wins in both modes. Size-only routing keeps results
        # independent of `num_threads`.
        if T === ComplexF32
            N >= LEAFFIRST_MIN && return false
        else
            (N >= _STOCKHAM_THREAD_MIN && N < LEAFFIRST_MIN) || return false
        end
    end
    return true
end
_use_stockham(::Type, N, num_threads) = false

function _stockham_stage(::Type{S}, R::Int, n::Int, D::Int, s::Int) where {S<:Real}
    m = n ÷ R
    W = _stockham_width(S)
    small = R == 4 && s < W && (m * s) % W == 0 && m * s >= W
    if small
        ms = m * s
        twr = Vector{S}(undef, 3 * ms)
        twi = Vector{S}(undef, 3 * ms)
        for j in 1:3, p in 0:m-1
            si, co = sincospi(2.0 * D * ((p * j) % n) / n)
            for q in 1:s
                twr[(j-1)*ms+p*s+q] = S(co)
                twi[(j-1)*ms+p*s+q] = S(si)
            end
        end
        return StockhamStage{S}(4, true, twr, twi)
    end
    twr = Vector{S}(undef, (R - 1) * m)
    twi = Vector{S}(undef, (R - 1) * m)
    for p in 0:m-1, j in 1:R-1
        si, co = sincospi(2.0 * D * ((p * j) % n) / n)
        twr[(R-1)*p+j] = S(co)
        twi[(R-1)*p+j] = S(si)
    end
    return StockhamStage{S}(R, false, twr, twi)
end

# Stage chains (twiddle tables included) are immutable and shared across all
# plans of one (element type, length, direction) for the session, FFTW-style:
# repeated planning — DSP-style code plans on every call — costs an allocation
# of the scratch planes and a dictionary hit instead of a sincospi sweep.
const _STOCKHAM_STAGE_LOCK = ReentrantLock()
const _STOCKHAM_STAGE_CACHE = Dict{Tuple{DataType,Int,Int},Any}()

function _stockham_stages_cached(::Type{S}, n::Int, D::Int) where {S<:Real}
    key = (S, n, D)
    r = lock(_STOCKHAM_STAGE_LOCK) do
        get!(() -> _stockham_build_stages(S, n, D), _STOCKHAM_STAGE_CACHE, key)
    end
    return r::Vector{StockhamStage{S}}
end

function StockhamChain{S}(n::Int, dir::Direction, nt::Int=1) where {S<:Real}
    stages = _stockham_stages_cached(S, n, direction_sign(dir))
    return StockhamChain{S}(n, stages,
                            Vector{S}(undef, n), Vector{S}(undef, n),
                            Vector{S}(undef, n), Vector{S}(undef, n), max(nt, 1))
end

function _stockham_build_stages(::Type{S}, n::Int, D::Int; s0::Int=1) where {S<:Real}
    a = trailing_zeros(n)
    r = n >> a
    ps = Pair{Int,Int}[]
    for p in STOCKHAM_RADICES
        c = 0
        while r % p == 0
            r ÷= p
            c += 1
        end
        c > 0 && push!(ps, p => c)
    end
    r == 1 || throw(ArgumentError("length $n is not smooth over $STOCKHAM_RADICES"))
    f = a == 0 ? 1 : isodd(a) ? (a >= 3 ? 8 : 2) : 4
    k4 = (a - trailing_zeros(f)) ÷ 2
    stages = StockhamStage{S}[]
    ncur = n
    s = s0
    for _ in 1:k4
        push!(stages, _stockham_stage(S, 4, ncur, D, s))
        s *= 4
        ncur >>= 2
    end
    for (p, c) in ps, _ in 1:c
        push!(stages, _stockham_stage(S, p, ncur, D, s))
        s *= p
        ncur ÷= p
    end
    f > 1 && push!(stages, _stockham_stage(S, f, ncur, D, s))
    return stages
end

# ---------------------------------------------------------------------------
# butterflies (split-plane values; scalar or SIMD.Vec — same code)
# ---------------------------------------------------------------------------
@inline _scmul(wr, wi, xr, xi) = (muladd(wr, xr, -wi * xi), muladd(wr, xi, wi * xr))
@inline function _sbf4(ar, ai, br, bi, cr, ci, dr, di, ::Val{D}) where {D}
    apc_r = ar + cr; apc_i = ai + ci
    amc_r = ar - cr; amc_i = ai - ci
    bpd_r = br + dr; bpd_i = bi + di
    bmd_r = br - dr; bmd_i = bi - di
    y0r = apc_r + bpd_r; y0i = apc_i + bpd_i
    y2r = apc_r - bpd_r; y2i = apc_i - bpd_i
    if D == -1
        y1r = amc_r + bmd_i; y1i = amc_i - bmd_r
        y3r = amc_r - bmd_i; y3i = amc_i + bmd_r
    else
        y1r = amc_r - bmd_i; y1i = amc_i + bmd_r
        y3r = amc_r + bmd_i; y3i = amc_i - bmd_r
    end
    return y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i
end

# generic odd-radix DFT on split planes (conjugate-symmetric pairing, as in
# odd_codelets.jl); coefficient tuples are compile-time constants
@generated function _sodd_consts(::Type{S}, ::Val{P}, ::Val{D}) where {S,P,D}
    h = (P - 1) >> 1
    cc = ntuple(i -> S(cospi(2 * ((((i - 1) % h + 1) * ((i - 1) ÷ h + 1)) % P) / P)), h * h)
    ss = ntuple(i -> S(D * sinpi(2 * ((((i - 1) % h + 1) * ((i - 1) ÷ h + 1)) % P) / P)), h * h)
    return :(($cc, $ss))
end
@generated function _svbroad(::Type{Vec{W,S}}, t::NTuple{N,S}) where {W,S,N}
    :(@inbounds $(Expr(:tuple, (:(Vec{W,S}(t[$i])) for i in 1:N)...)))
end
@generated function _sbfp(xr::NTX, xi::NTX, cc::NTC, ss::NTC) where {NTX<:Tuple,NTC<:Tuple}
    P = length(NTX.parameters)
    h = (P - 1) >> 1
    body = Expr[]
    for j in 1:h
        push!(body, :($(Symbol(:t, j, :r)) = xr[$(j + 1)] + xr[$(P + 1 - j)]),
                    :($(Symbol(:t, j, :i)) = xi[$(j + 1)] + xi[$(P + 1 - j)]),
                    :($(Symbol(:d, j, :r)) = xr[$(j + 1)] - xr[$(P + 1 - j)]),
                    :($(Symbol(:d, j, :i)) = xi[$(j + 1)] - xi[$(P + 1 - j)]))
    end
    sr = :(xr[1]); si = :(xi[1])
    for j in 1:h
        sr = :($sr + $(Symbol(:t, j, :r)))
        si = :($si + $(Symbol(:t, j, :i)))
    end
    push!(body, :(y0r = $sr), :(y0i = $si))
    for k in 1:h
        ur = :(xr[1]); ui = :(xi[1])
        for j in 1:h
            c = :(cc[$((k - 1) * h + j)])
            ur = :(muladd($c, $(Symbol(:t, j, :r)), $ur))
            ui = :(muladd($c, $(Symbol(:t, j, :i)), $ui))
        end
        vr = :(ss[$((k - 1) * h + 1)] * $(Symbol(:d, 1, :r)))
        vi = :(ss[$((k - 1) * h + 1)] * $(Symbol(:d, 1, :i)))
        for j in 2:h
            sx = :(ss[$((k - 1) * h + j)])
            vr = :(muladd($sx, $(Symbol(:d, j, :r)), $vr))
            vi = :(muladd($sx, $(Symbol(:d, j, :i)), $vi))
        end
        push!(body, :($(Symbol(:u, k, :r)) = $ur), :($(Symbol(:u, k, :i)) = $ui),
                    :($(Symbol(:v, k, :r)) = $vr), :($(Symbol(:v, k, :i)) = $vi))
    end
    outs = Any[:y0r, :y0i]
    for k in 1:h
        push!(outs, :($(Symbol(:u, k, :r)) - $(Symbol(:v, k, :i))), :($(Symbol(:u, k, :i)) + $(Symbol(:v, k, :r))))
    end
    for k in h:-1:1
        push!(outs, :($(Symbol(:u, k, :r)) + $(Symbol(:v, k, :i))), :($(Symbol(:u, k, :i)) - $(Symbol(:v, k, :r))))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(body...)
            tuple($(outs...))
        end
    end
end
@inline _sstore!(a, i, v::Vec) = _sst!(a, i, v)
@inline _sstore!(a, i, v) = _sst!(a, i, v)
@generated function _sodd_store!(yre, yim, ys::NT, wr::NTW, wi::NTW, yb::Int, s::Int, q::Int) where {NT<:Tuple,NTW<:Tuple}
    M = length(NTW.parameters)
    exs = [quote
               or, oi = _scmul(wr[$j], wi[$j], ys[$(2j + 1)], ys[$(2j + 2)])
               _sstore!(yre, yb + $j * s + q, or)
               _sstore!(yim, yb + $j * s + q, oi)
           end for j in 1:M]
    quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(exs...)
        end
        return nothing
    end
end

# lane interleaves for the small-stride radix-4 stage
@generated function _sileave2(u::Vec{W,S}, v::Vec{W,S}, ::Val{G}) where {W,S,G}
    B = W ÷ G
    idx(g, off) = (iseven(g) ? 0 : W) + (g >> 1) * G + off
    lo = ntuple(i -> idx((i - 1) ÷ G, (i - 1) % G), W)
    hi = ntuple(i -> idx((i - 1) ÷ G + B, (i - 1) % G), W)
    :((shufflevector(u, v, Val($lo)), shufflevector(u, v, Val($hi))))
end
@inline function _sileave4(y0::Vec{W,S}, y1::Vec{W,S}, y2::Vec{W,S}, y3::Vec{W,S}, ::Val{G}) where {W,S,G}
    a0, a1 = _sileave2(y0, y1, Val(G))
    b0, b1 = _sileave2(y2, y3, Val(G))
    o0, o1 = _sileave2(a0, b0, Val(2G))
    o2, o3 = _sileave2(a1, b1, Val(2G))
    return o0, o1, o2, o3
end
@generated function _sdeint2(u::Vec{W,S}, v::Vec{W,S}) where {W,S}
    even = ntuple(i -> 2i - 2, W)
    odd = ntuple(i -> 2i - 1, W)
    :((shufflevector(u, v, Val($even)), shufflevector(u, v, Val($odd))))
end
@generated function _sint2(r::Vec{W,S}, i::Vec{W,S}) where {W,S}
    lo = ntuple(j -> isodd(j) ? (j - 1) >> 1 : W + ((j - 1) >> 1), W)
    hi = ntuple(j -> isodd(j) ? (W >> 1) + ((j - 1) >> 1) : W + (W >> 1) + ((j - 1) >> 1), W)
    :((shufflevector(r, i, Val($lo)), shufflevector(r, i, Val($hi))))
end

# ---------------------------------------------------------------------------
# stage loops (x → y planes; overlap windows cover strides that are not
# multiples of W: recomputed lanes store identical values)
# ---------------------------------------------------------------------------
function _stage4!(yre::Vector{S}, yim::Vector{S}, xre::Vector{S}, xim::Vector{S},
                  n::Int, s::Int, tw::StockhamStage{S}, ::Val{D}, lo::Int, hi::Int,
                  qlo::Int=1, qhi::Int=(s < _stockham_width(S) ? 1 : cld(s, _stockham_width(S)))) where {S,D}
    W = _stockham_width(S)
    if tw.small
        # `lo:hi` are W-blocks of the lane-expanded twiddle layout here
        s == 1 && return _stage4_small!(yre, yim, xre, xim, n, Val(1), tw, Val(D), lo, hi)
        s == 2 && return _stage4_small!(yre, yim, xre, xim, n, Val(2), tw, Val(D), lo, hi)
        return _stage4_small!(yre, yim, xre, xim, n, Val(4), tw, Val(D), lo, hi)
    end
    m = n >> 2
    ms = m * s
    twr = tw.twr; twi = tw.twi
    if s < W
        @inbounds for p in lo-1:hi-1
            w1r = _sget(twr, 3p+1); w1i = _sget(twi, 3p+1); w2r = _sget(twr, 3p+2); w2i = _sget(twi, 3p+2); w3r = _sget(twr, 3p+3); w3i = _sget(twi, 3p+3)
            xb = s * p; yb = 4s * p
            for q in 1:s
                y0r, y0i, t1r, t1i, t2r, t2i, t3r, t3i = _sbf4(
                    _sget(xre, xb+q), _sget(xim, xb+q), _sget(xre, xb+ms+q), _sget(xim, xb+ms+q),
                    _sget(xre, xb+2ms+q), _sget(xim, xb+2ms+q), _sget(xre, xb+3ms+q), _sget(xim, xb+3ms+q), Val(D))
                y1r, y1i = _scmul(w1r, w1i, t1r, t1i)
                y2r, y2i = _scmul(w2r, w2i, t2r, t2i)
                y3r, y3i = _scmul(w3r, w3i, t3r, t3i)
                _sst!(yre, yb+q, y0r);    _sst!(yim, yb+q, y0i)
                _sst!(yre, yb+s+q, y1r);  _sst!(yim, yb+s+q, y1i)
                _sst!(yre, yb+2s+q, y2r); _sst!(yim, yb+2s+q, y2i)
                _sst!(yre, yb+3s+q, y3r); _sst!(yim, yb+3s+q, y3i)
            end
        end
        return nothing
    end
    V = Vec{W,S}
    @inbounds for p in lo-1:hi-1
        w1r = V(_sget(twr, 3p+1)); w1i = V(_sget(twi, 3p+1)); w2r = V(_sget(twr, 3p+2)); w2i = V(_sget(twi, 3p+2)); w3r = V(_sget(twr, 3p+3)); w3i = V(_sget(twi, 3p+3))
        xb = s * p; yb = 4s * p
        for b in qlo:qhi
            q0 = (b - 1) * W + 1
            q = q0 + W - 1 <= s ? q0 : s - W + 1
            ar = _sld(V, xre, xb + q);        ai = _sld(V, xim, xb + q)
            br = _sld(V, xre, xb + ms + q);   bi = _sld(V, xim, xb + ms + q)
            cr = _sld(V, xre, xb + 2ms + q);  ci = _sld(V, xim, xb + 2ms + q)
            dr = _sld(V, xre, xb + 3ms + q);  di = _sld(V, xim, xb + 3ms + q)
            y0r, y0i, t1r, t1i, t2r, t2i, t3r, t3i = _sbf4(ar, ai, br, bi, cr, ci, dr, di, Val(D))
            y1r, y1i = _scmul(w1r, w1i, t1r, t1i)
            y2r, y2i = _scmul(w2r, w2i, t2r, t2i)
            y3r, y3i = _scmul(w3r, w3i, t3r, t3i)
            _sst!(yre, yb + q, y0r);      _sst!(yim, yb + q, y0i)
            _sst!(yre, yb + s + q, y1r);  _sst!(yim, yb + s + q, y1i)
            _sst!(yre, yb + 2s + q, y2r); _sst!(yim, yb + 2s + q, y2i)
            _sst!(yre, yb + 3s + q, y3r); _sst!(yim, yb + 3s + q, y3i)
        end
    end
    return nothing
end

function _stage4_small!(yre::Vector{S}, yim::Vector{S}, xre, xim, n::Int, ::Val{G}, tw::StockhamStage{S}, ::Val{D}, lo::Int, hi::Int) where {S,G,D}
    W = _stockham_width(S)
    m = n >> 2
    ms = m * G
    twr = tw.twr; twi = tw.twi
    V = Vec{W,S}
    @inbounds for b in lo:hi
        f = (b - 1) * W + 1
        w1r = _sld(V, twr, f);       w1i = _sld(V, twi, f)
        w2r = _sld(V, twr, ms + f);  w2i = _sld(V, twi, ms + f)
        w3r = _sld(V, twr, 2ms + f); w3i = _sld(V, twi, 2ms + f)
        ar = _sld(V, xre, f);        ai = _sld(V, xim, f)
        br = _sld(V, xre, ms + f);   bi = _sld(V, xim, ms + f)
        cr = _sld(V, xre, 2ms + f);  ci = _sld(V, xim, 2ms + f)
        dr = _sld(V, xre, 3ms + f);  di = _sld(V, xim, 3ms + f)
        y0r, y0i, t1r, t1i, t2r, t2i, t3r, t3i = _sbf4(ar, ai, br, bi, cr, ci, dr, di, Val(D))
        y1r, y1i = _scmul(w1r, w1i, t1r, t1i)
        y2r, y2i = _scmul(w2r, w2i, t2r, t2i)
        y3r, y3i = _scmul(w3r, w3i, t3r, t3i)
        o0r, o1r, o2r, o3r = _sileave4(y0r, y1r, y2r, y3r, Val(G))
        o0i, o1i, o2i, o3i = _sileave4(y0i, y1i, y2i, y3i, Val(G))
        base = 4 * (f - 1)
        _sst!(yre, base + 1, o0r);      _sst!(yim, base + 1, o0i)
        _sst!(yre, base + W + 1, o1r);  _sst!(yim, base + W + 1, o1i)
        _sst!(yre, base + 2W + 1, o2r); _sst!(yim, base + 2W + 1, o2i)
        _sst!(yre, base + 3W + 1, o3r); _sst!(yim, base + 3W + 1, o3i)
    end
    return nothing
end

function _stage2!(yre::Vector{S}, yim::Vector{S}, xre, xim, s::Int, lo::Int, hi::Int) where {S}
    W = _stockham_width(S)
    if s < W
        @inbounds for q in 1:s
            ar = _sget(xre, q); ai = _sget(xim, q); br = _sget(xre, s+q); bi = _sget(xim, s+q)
            _sst!(yre, q, ar + br);   _sst!(yim, q, ai + bi)
            _sst!(yre, s+q, ar - br); _sst!(yim, s+q, ai - bi)
        end
        return nothing
    end
    V = Vec{W,S}
    @inbounds for b in lo:hi
        q0 = (b - 1) * W + 1
        q = q0 + W - 1 <= s ? q0 : s - W + 1
        ar = _sld(V, xre, q); ai = _sld(V, xim, q)
        br = _sld(V, xre, s + q); bi = _sld(V, xim, s + q)
        _sst!(yre, q, ar + br);     _sst!(yim, q, ai + bi)
        _sst!(yre, s + q, ar - br); _sst!(yim, s + q, ai - bi)
    end
    return nothing
end

@inline function _bf8_tail(o, c, ::Val{D}) where {D}
    if D == -1
        return (c * (o[3] + o[4]), c * (o[4] - o[3]), o[6], -o[5], c * (o[8] - o[7]), -c * (o[7] + o[8]))
    else
        return (c * (o[3] - o[4]), c * (o[4] + o[3]), -o[6], o[5], -c * (o[7] + o[8]), c * (o[7] - o[8]))
    end
end
function _stage8!(yre::Vector{S}, yim::Vector{S}, xre, xim, s::Int, ::Val{D}, lo::Int, hi::Int) where {S,D}
    W = _stockham_width(S)
    if s < W
        c = S(sqrt(0.5))
        @inbounds for q in 1:s
            e = _sbf4(_sget(xre, q), _sget(xim, q), _sget(xre, 2s+q), _sget(xim, 2s+q), _sget(xre, 4s+q), _sget(xim, 4s+q), _sget(xre, 6s+q), _sget(xim, 6s+q), Val(D))
            o = _sbf4(_sget(xre, s+q), _sget(xim, s+q), _sget(xre, 3s+q), _sget(xim, 3s+q), _sget(xre, 5s+q), _sget(xim, 5s+q), _sget(xre, 7s+q), _sget(xim, 7s+q), Val(D))
            t1r, t1i, t2r, t2i, t3r, t3i = _bf8_tail(o, c, Val(D))
            _sst!(yre, q, e[1] + o[1]);      _sst!(yim, q, e[2] + o[2])
            _sst!(yre, s+q, e[3] + t1r);     _sst!(yim, s+q, e[4] + t1i)
            _sst!(yre, 2s+q, e[5] + t2r);    _sst!(yim, 2s+q, e[6] + t2i)
            _sst!(yre, 3s+q, e[7] + t3r);    _sst!(yim, 3s+q, e[8] + t3i)
            _sst!(yre, 4s+q, e[1] - o[1]);   _sst!(yim, 4s+q, e[2] - o[2])
            _sst!(yre, 5s+q, e[3] - t1r);    _sst!(yim, 5s+q, e[4] - t1i)
            _sst!(yre, 6s+q, e[5] - t2r);    _sst!(yim, 6s+q, e[6] - t2i)
            _sst!(yre, 7s+q, e[7] - t3r);    _sst!(yim, 7s+q, e[8] - t3i)
        end
        return nothing
    end
    V = Vec{W,S}
    c = V(S(sqrt(0.5)))
    @inbounds for b in lo:hi
        q0 = (b - 1) * W + 1
        q = q0 + W - 1 <= s ? q0 : s - W + 1
        e = _sbf4(_sld(V, xre, q), _sld(V, xim, q), _sld(V, xre, 2s + q), _sld(V, xim, 2s + q),
                  _sld(V, xre, 4s + q), _sld(V, xim, 4s + q), _sld(V, xre, 6s + q), _sld(V, xim, 6s + q), Val(D))
        o = _sbf4(_sld(V, xre, s + q), _sld(V, xim, s + q), _sld(V, xre, 3s + q), _sld(V, xim, 3s + q),
                  _sld(V, xre, 5s + q), _sld(V, xim, 5s + q), _sld(V, xre, 7s + q), _sld(V, xim, 7s + q), Val(D))
        t1r, t1i, t2r, t2i, t3r, t3i = _bf8_tail(o, c, Val(D))
        _sst!(yre, q, e[1] + o[1]);      _sst!(yim, q, e[2] + o[2])
        _sst!(yre, s + q, e[3] + t1r);   _sst!(yim, s + q, e[4] + t1i)
        _sst!(yre, 2s + q, e[5] + t2r);  _sst!(yim, 2s + q, e[6] + t2i)
        _sst!(yre, 3s + q, e[7] + t3r);  _sst!(yim, 3s + q, e[8] + t3i)
        _sst!(yre, 4s + q, e[1] - o[1]); _sst!(yim, 4s + q, e[2] - o[2])
        _sst!(yre, 5s + q, e[3] - t1r);  _sst!(yim, 5s + q, e[4] - t1i)
        _sst!(yre, 6s + q, e[5] - t2r);  _sst!(yim, 6s + q, e[6] - t2i)
        _sst!(yre, 7s + q, e[7] - t3r);  _sst!(yim, 7s + q, e[8] - t3i)
    end
    return nothing
end

function _stagep!(yre::Vector{S}, yim::Vector{S}, xre, xim, n::Int, s::Int,
                  tw::StockhamStage{S}, ::Val{P}, ::Val{D}, lo::Int, hi::Int,
                  qlo::Int=1, qhi::Int=(s < _stockham_width(S) ? 1 : cld(s, _stockham_width(S)))) where {S,P,D}
    m = n ÷ P
    ms = m * s
    twr = tw.twr; twi = tw.twi
    cc, ss = _sodd_consts(S, Val(P), Val(D))
    W = _stockham_width(S)
    if s >= W
        V = Vec{W,S}
        ccv = _svbroad(V, cc)
        ssv = _svbroad(V, ss)
        @inbounds for p in lo-1:hi-1
            wr = ntuple(j -> V(@inbounds _sget(twr, (P-1)*p+j)), Val(P - 1))
            wi = ntuple(j -> V(@inbounds _sget(twi, (P-1)*p+j)), Val(P - 1))
            xb = s * p; yb = P * s * p
            for b in qlo:qhi
                q = (b - 1) * W + 1
                qq = q + W - 1 <= s ? q : s - W + 1
                xsr = ntuple(j -> @inbounds(_sld(V, xre, xb + (j - 1) * ms + qq)), Val(P))
                xsi = ntuple(j -> @inbounds(_sld(V, xim, xb + (j - 1) * ms + qq)), Val(P))
                ys = _sbfp(xsr, xsi, ccv, ssv)
                _sst!(yre, yb + qq, ys[1])
                _sst!(yim, yb + qq, ys[2])
                _sodd_store!(yre, yim, ys, wr, wi, yb, s, qq)
            end
        end
    else
        @inbounds for p in lo-1:hi-1
            wr = ntuple(j -> @inbounds(_sget(twr, (P-1)*p+j)), Val(P - 1))
            wi = ntuple(j -> @inbounds(_sget(twi, (P-1)*p+j)), Val(P - 1))
            xb = s * p; yb = P * s * p
            for q in 1:s
                xsr = ntuple(j -> @inbounds(_sget(xre, xb+(j-1)*ms+q)), Val(P))
                xsi = ntuple(j -> @inbounds(_sget(xim, xb+(j-1)*ms+q)), Val(P))
                ys = _sbfp(xsr, xsi, cc, ss)
                _sst!(yre, yb+q, ys[1])
                _sst!(yim, yb+q, ys[2])
                _sodd_store!(yre, yim, ys, wr, wi, yb, s, q)
            end
        end
    end
    return nothing
end


# ---------------------------------------------------------------------------
# fused edges: the first small radix-4 stage can read the interleaved complex
# input directly (deinterleave in registers), and a twiddle-free final stage
# can write interleaved complex output — each saves one full conversion pass.
# ---------------------------------------------------------------------------
function _stage4_first_fused!(yre::Vector{S}, yim::Vector{S}, x::Vector{Complex{S}}, start_in::Int,
                              n::Int, tw::StockhamStage{S}, ::Val{D}, lo::Int, hi::Int) where {S,D}
    W = _stockham_width(S)
    m = n >> 2
    twr = tw.twr; twi = tw.twi
    V = Vec{W,S}
    sz = sizeof(S)
    GC.@preserve x begin
        px = Ptr{S}(pointer(x)) + 2sz * (start_in - 1)
        @inbounds for b in lo:hi
            f = (b - 1) * W + 1
            b0 = 2 * (f - 1)
            u = vload(V, px + sz * b0);            v = vload(V, px + sz * (b0 + W))
            ar, ai = _sdeint2(u, v)
            u = vload(V, px + sz * (b0 + 2m));     v = vload(V, px + sz * (b0 + 2m + W))
            br, bi = _sdeint2(u, v)
            u = vload(V, px + sz * (b0 + 4m));     v = vload(V, px + sz * (b0 + 4m + W))
            cr, ci = _sdeint2(u, v)
            u = vload(V, px + sz * (b0 + 6m));     v = vload(V, px + sz * (b0 + 6m + W))
            dr, di = _sdeint2(u, v)
            w1r = _sld(V, twr, f);       w1i = _sld(V, twi, f)
            w2r = _sld(V, twr, m + f);   w2i = _sld(V, twi, m + f)
            w3r = _sld(V, twr, 2m + f);  w3i = _sld(V, twi, 2m + f)
            y0r, y0i, t1r, t1i, t2r, t2i, t3r, t3i = _sbf4(ar, ai, br, bi, cr, ci, dr, di, Val(D))
            y1r, y1i = _scmul(w1r, w1i, t1r, t1i)
            y2r, y2i = _scmul(w2r, w2i, t2r, t2i)
            y3r, y3i = _scmul(w3r, w3i, t3r, t3i)
            o0r, o1r, o2r, o3r = _sileave4(y0r, y1r, y2r, y3r, Val(1))
            o0i, o1i, o2i, o3i = _sileave4(y0i, y1i, y2i, y3i, Val(1))
            base = 4 * (f - 1)
            _sst!(yre, base + 1, o0r);      _sst!(yim, base + 1, o0i)
            _sst!(yre, base + W + 1, o1r);  _sst!(yim, base + W + 1, o1i)
            _sst!(yre, base + 2W + 1, o2r); _sst!(yim, base + 2W + 1, o2i)
            _sst!(yre, base + 3W + 1, o3r); _sst!(yim, base + 3W + 1, o3i)
        end
    end
    return nothing
end

# twiddle-free final stages writing interleaved complex output
function _stage4_last_fused!(y::Vector{Complex{S}}, start_out::Int, xre::Vector{S}, xim::Vector{S},
                             s::Int, ::Val{D}, lo::Int, hi::Int) where {S,D}
    W = _stockham_width(S)
    V = Vec{W,S}
    sz = sizeof(S)
    GC.@preserve y begin
        py = Ptr{S}(pointer(y)) + 2sz * (start_out - 1)
        @inbounds for b in lo:hi
            q0 = (b - 1) * W + 1
            q = q0 + W - 1 <= s ? q0 : s - W + 1
            e = _sbf4(_sld(V, xre, q), _sld(V, xim, q), _sld(V, xre, s + q), _sld(V, xim, s + q),
                      _sld(V, xre, 2s + q), _sld(V, xim, 2s + q), _sld(V, xre, 3s + q), _sld(V, xim, 3s + q), Val(D))
            Base.Cartesian.@nexprs 4 j -> begin
                lo, hi = _sint2(e[2j-1], e[2j])
                c0 = (j - 1) * s + q - 1
                vstore(lo, py + sz * 2c0)
                vstore(hi, py + sz * (2c0 + W))
            end
        end
    end
    return nothing
end
function _stage8_last_fused!(y::Vector{Complex{S}}, start_out::Int, xre::Vector{S}, xim::Vector{S},
                             s::Int, ::Val{D}, lo::Int, hi::Int) where {S,D}
    W = _stockham_width(S)
    V = Vec{W,S}
    c = V(S(sqrt(0.5)))
    sz = sizeof(S)
    GC.@preserve y begin
        py = Ptr{S}(pointer(y)) + 2sz * (start_out - 1)
        @inbounds for b in lo:hi
            q0 = (b - 1) * W + 1
            q = q0 + W - 1 <= s ? q0 : s - W + 1
            e = _sbf4(_sld(V, xre, q), _sld(V, xim, q), _sld(V, xre, 2s + q), _sld(V, xim, 2s + q),
                      _sld(V, xre, 4s + q), _sld(V, xim, 4s + q), _sld(V, xre, 6s + q), _sld(V, xim, 6s + q), Val(D))
            o = _sbf4(_sld(V, xre, s + q), _sld(V, xim, s + q), _sld(V, xre, 3s + q), _sld(V, xim, 3s + q),
                      _sld(V, xre, 5s + q), _sld(V, xim, 5s + q), _sld(V, xre, 7s + q), _sld(V, xim, 7s + q), Val(D))
            t1r, t1i, t2r, t2i, t3r, t3i = _bf8_tail(o, c, Val(D))
            ys = (e[1] + o[1], e[2] + o[2], e[3] + t1r, e[4] + t1i, e[5] + t2r, e[6] + t2i, e[7] + t3r, e[8] + t3i,
                  e[1] - o[1], e[2] - o[2], e[3] - t1r, e[4] - t1i, e[5] - t2r, e[6] - t2i, e[7] - t3r, e[8] - t3i)
            Base.Cartesian.@nexprs 8 j -> begin
                lo, hi = _sint2(ys[2j-1], ys[2j])
                c0 = (j - 1) * s + q - 1
                vstore(lo, py + sz * 2c0)
                vstore(hi, py + sz * (2c0 + W))
            end
        end
    end
    return nothing
end
function _stage2_last_fused!(y::Vector{Complex{S}}, start_out::Int, xre::Vector{S}, xim::Vector{S}, s::Int, lo::Int, hi::Int) where {S}
    W = _stockham_width(S)
    V = Vec{W,S}
    sz = sizeof(S)
    GC.@preserve y begin
        py = Ptr{S}(pointer(y)) + 2sz * (start_out - 1)
        @inbounds for b in lo:hi
            q0 = (b - 1) * W + 1
            q = q0 + W - 1 <= s ? q0 : s - W + 1
            ar = _sld(V, xre, q); ai = _sld(V, xim, q)
            br = _sld(V, xre, s + q); bi = _sld(V, xim, s + q)
            lo, hi = _sint2(ar + br, ai + bi)
            vstore(lo, py + sz * 2 * (q - 1)); vstore(hi, py + sz * (2 * (q - 1) + W))
            lo, hi = _sint2(ar - br, ai - bi)
            c0 = s + q - 1
            vstore(lo, py + sz * 2c0); vstore(hi, py + sz * (2c0 + W))
        end
    end
    return nothing
end
# ---------------------------------------------------------------------------
# interleaved-complex edges and the driver
# ---------------------------------------------------------------------------
function _sdeint_all!(are::Vector{S}, aim::Vector{S}, x::AbstractVector{Complex{S}}, start_in::Int, n::Int) where {S}
    W = _stockham_width(S)
    if x isa Vector{Complex{S}} && n >= W
        sz = sizeof(S)
        GC.@preserve x begin
            px = Ptr{S}(pointer(x)) + 2sz * (start_in - 1)
            @inbounds for c0 in 1:W:n
                c = c0 + W - 1 <= n ? c0 : n - W + 1
                u = vload(Vec{W,S}, px + sz * 2 * (c - 1))
                v = vload(Vec{W,S}, px + sz * (2 * (c - 1) + W))
                r, i = _sdeint2(u, v)
                vstore(r, are, c)
                vstore(i, aim, c)
            end
        end
    else
        @inbounds for c in 1:n
            z = x[start_in+c-1]
            are[c] = real(z)
            aim[c] = imag(z)
        end
    end
    return nothing
end
function _sint_all!(y::AbstractVector{Complex{S}}, start_out::Int, are::Vector{S}, aim::Vector{S}, n::Int) where {S}
    W = _stockham_width(S)
    if y isa Vector{Complex{S}} && n >= W
        sz = sizeof(S)
        GC.@preserve y begin
            py = Ptr{S}(pointer(y)) + 2sz * (start_out - 1)
            @inbounds for c0 in 1:W:n
                c = c0 + W - 1 <= n ? c0 : n - W + 1
                r = vload(Vec{W,S}, are, c)
                i = vload(Vec{W,S}, aim, c)
                lo, hi = _sint2(r, i)
                vstore(lo, py + sz * 2 * (c - 1))
                vstore(hi, py + sz * (2 * (c - 1) + W))
            end
        end
    else
        @inbounds for c in 1:n
            y[start_out+c-1] = Complex{S}(are[c], aim[c])
        end
    end
    return nothing
end

function _sdeint_blocks!(are::Vector{S}, aim::Vector{S}, x::Vector{Complex{S}}, start_in::Int,
                         n::Int, lo::Int, hi::Int) where {S}
    W = _stockham_width(S)
    sz = sizeof(S)
    V = Vec{W,S}
    GC.@preserve x begin
        px = Ptr{S}(pointer(x)) + 2sz * (start_in - 1)
        @inbounds for b in lo:hi
            c0 = (b - 1) * W + 1
            c = c0 + W - 1 <= n ? c0 : n - W + 1
            u = vload(V, px + sz * 2 * (c - 1))
            v = vload(V, px + sz * (2 * (c - 1) + W))
            r, i = _sdeint2(u, v)
            _sst!(are, c, r)
            _sst!(aim, c, i)
        end
    end
    return nothing
end
function _sint_blocks!(y::Vector{Complex{S}}, start_out::Int, are::Vector{S}, aim::Vector{S},
                       n::Int, lo::Int, hi::Int) where {S}
    W = _stockham_width(S)
    sz = sizeof(S)
    V = Vec{W,S}
    GC.@preserve y begin
        py = Ptr{S}(pointer(y)) + 2sz * (start_out - 1)
        @inbounds for b in lo:hi
            c0 = (b - 1) * W + 1
            c = c0 + W - 1 <= n ? c0 : n - W + 1
            r = _sld(V, are, c)
            i = _sld(V, aim, c)
            l, h = _sint2(r, i)
            vstore(l, py + sz * 2 * (c - 1))
            vstore(h, py + sz * (2 * (c - 1) + W))
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)
Run the chain: `out[start_out .+ (0:n-1)] = FFT(in[start_in .+ (0:n-1)])`.
The input may be real or complex; unit stride. With `ch.nt > 1` and
`n >= _STOCKHAM_THREAD_MIN`, a task team runs every stage in W-aligned
chunks with a barrier between stages — results are bitwise identical to
the serial order.
"""
function _stockham_exec!(out::AbstractVector{Complex{S}}, in::AbstractVector, start_out::Int, start_in::Int,
                         d::Direction, ch::StockhamChain{S}) where {S}
    D = direction_sign(d)
    nt = (ch.nt > 1 && ch.n >= _STOCKHAM_THREAD_MIN) ? min(ch.nt, Threads.nthreads()) : 1
    if nt == 1
        _stockham_run!(out, in, start_out, start_in, D, ch, 1, 1, nothing)
    else
        bar = _SBarrier(0, 0, nt)
        Base.@sync for c in 1:nt
            Threads.@spawn _stockham_run!(out, in, start_out, start_in, D, ch, c, nt, bar)
        end
    end
    return nothing
end

function _stockham_run!(out::AbstractVector{Complex{S}}, in::AbstractVector, start_out::Int, start_in::Int,
                        D::Int, ch::StockhamChain{S}, c::Int, nt::Int,
                        bar::Union{Nothing,_SBarrier}) where {S}
    n = ch.n
    W = _stockham_width(S)
    xr, xi, yr, yi = ch.are, ch.aim, ch.bre, ch.bim
    nstages = length(ch.stages)
    # fuse the deinterleave into a first small radix-4 stage when possible
    first_fused = in isa Vector{Complex{S}} && nstages > 1 && ch.stages[1].R == 4 &&
                  ch.stages[1].small && (n >> 2) % W == 0 && n >> 2 >= W
    # fuse the interleave into a twiddle-free final pow2 stage when possible
    laststage = ch.stages[nstages]
    last_fused = out isa Vector{Complex{S}} && nstages > 1 &&
                 (laststage.R == 2 || laststage.R == 4 || laststage.R == 8) &&
                 !laststage.small && n ÷ laststage.R >= W
    Vd = D == -1 ? Val(-1) : Val(1)
    s = 1
    ncur = n
    if first_fused
        lo, hi = _srange(c, nt, (n >> 2) ÷ W)
        hi >= lo && _stage4_first_fused!(yr, yi, in::Vector{Complex{S}}, start_in, n, ch.stages[1], Vd, lo, hi)
        _swait(bar)
        s = 4
        ncur >>= 2
        xr, xi, yr, yi = yr, yi, xr, xi
    elseif in isa AbstractVector{Complex{S}}
        if in isa Vector{Complex{S}} && n >= W && (n % W == 0 || cld(n, W) >= 2 * nt)
            lo, hi = _srange(c, nt, cld(n, W))
            hi >= lo && _sdeint_blocks!(xr, xi, in, start_in, n, lo, hi)
        elseif c == 1
            _sdeint_all!(xr, xi, in, start_in, n)
        end
        _swait(bar)
    else
        lo, hi = _srange(c, nt, n)
        @inbounds for e in lo:hi
            xr[e] = S(real(in[start_in+e-1]))
            xi[e] = S(imag(in[start_in+e-1]))
        end
        _swait(bar)
    end
    for k in (first_fused ? 2 : 1):(last_fused ? nstages - 1 : nstages)
        tw = ch.stages[k]
        R = tw.R
        if R == 4
            if tw.small
                lo, hi = _srange(c, nt, ((ncur >> 2) * s) ÷ W)
                hi >= lo && _stage4!(yr, yi, xr, xi, ncur, s, tw, Vd, lo, hi)
            else
                lo, hi, qlo, qhi = _spq(c, nt, ncur >> 2, s, W)
                hi >= lo && qhi >= qlo && _stage4!(yr, yi, xr, xi, ncur, s, tw, Vd, lo, hi, qlo, qhi)
            end
        elseif R == 2
            nb = s < W ? 1 : cld(s, W)
            lo, hi = (s >= W && (s % W == 0 || nb >= 2 * nt)) ? _srange(c, nt, nb) : (c == 1 ? (1, nb) : (1, 0))
            hi >= lo && _stage2!(yr, yi, xr, xi, s, lo, hi)
        elseif R == 8
            nb = s < W ? 1 : cld(s, W)
            lo, hi = (s >= W && (s % W == 0 || nb >= 2 * nt)) ? _srange(c, nt, nb) : (c == 1 ? (1, nb) : (1, 0))
            hi >= lo && _stage8!(yr, yi, xr, xi, s, Vd, lo, hi)
        elseif R == 3
            lo, hi, qlo, qhi = _spq(c, nt, ncur ÷ 3, s, W)
            hi >= lo && qhi >= qlo && _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(3), D, lo, hi, qlo, qhi)
        elseif R == 5
            lo, hi, qlo, qhi = _spq(c, nt, ncur ÷ 5, s, W)
            hi >= lo && qhi >= qlo && _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(5), D, lo, hi, qlo, qhi)
        elseif R == 7
            lo, hi, qlo, qhi = _spq(c, nt, ncur ÷ 7, s, W)
            hi >= lo && qhi >= qlo && _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(7), D, lo, hi, qlo, qhi)
        elseif R == 11
            lo, hi, qlo, qhi = _spq(c, nt, ncur ÷ 11, s, W)
            hi >= lo && qhi >= qlo && _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(11), D, lo, hi, qlo, qhi)
        else
            lo, hi, qlo, qhi = _spq(c, nt, ncur ÷ 13, s, W)
            hi >= lo && qhi >= qlo && _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(13), D, lo, hi, qlo, qhi)
        end
        _swait(bar)
        s *= R
        ncur ÷= R
        xr, xi, yr, yi = yr, yi, xr, xi
    end
    if last_fused
        yv = out::Vector{Complex{S}}
        nb = cld(s, W)
        lo, hi = (s % W == 0 || nb >= 2 * nt) ? _srange(c, nt, nb) : (c == 1 ? (1, nb) : (1, 0))
        if hi >= lo
            if laststage.R == 4
                _stage4_last_fused!(yv, start_out, xr, xi, s, Vd, lo, hi)
            elseif laststage.R == 8
                _stage8_last_fused!(yv, start_out, xr, xi, s, Vd, lo, hi)
            else
                _stage2_last_fused!(yv, start_out, xr, xi, s, lo, hi)
            end
        end
    elseif out isa Vector{Complex{S}} && n >= W && (n % W == 0 || cld(n, W) >= 2 * nt)
        lo, hi = _srange(c, nt, cld(n, W))
        hi >= lo && _sint_blocks!(out, start_out, xr, xi, n, lo, hi)
    elseif c == 1
        _sint_all!(out, start_out, xr, xi, n)
    end
    return nothing
end
@inline _stagep_dir!(yr, yi, xr, xi, n, s, tw, vp, D, lo, hi, qlo, qhi) =
    D == -1 ? _stagep!(yr, yi, xr, xi, n, s, tw, vp, Val(-1), lo, hi, qlo, qhi) :
    _stagep!(yr, yi, xr, xi, n, s, tw, vp, Val(1), lo, hi, qlo, qhi)

# ---------------------------------------------------------------------------
# batched pencils: `B` adjacent contiguous pencils run through the same stage
# kernels at once by building the chain with initial stride `B` — every lane
# block holds `B` pencils, so gathers/scatters stream contiguous runs and the
# butterflies vectorise across pencils. (Used for dims > 1 of N-d arrays.)
# ---------------------------------------------------------------------------
const _STOCKHAM_BATCH_LOCK = ReentrantLock()
const _STOCKHAM_BATCH_CACHE = Dict{Tuple{DataType,Int,Int,Int},Any}()
function _stockham_batch_stages(::Type{S}, n::Int, D::Int, B::Int) where {S<:Real}
    key = (S, n, D, B)
    r = lock(_STOCKHAM_BATCH_LOCK) do
        get!(() -> _stockham_build_stages(S, n, D; s0=B), _STOCKHAM_BATCH_CACHE, key)
    end
    return r::Vector{StockhamStage{S}}
end

"transform `B` pencils `avin[base + (b-1) + (j-1)*st]`, `b in 1:B`, into the same slots of `avout`"
function _stockham_pencil_batch!(avout::AbstractVector{Complex{S}}, avin::AbstractVector{Complex{S}},
                                 base::Int, st::Int, B::Int,
                                 D::Int, n::Int, stages::Vector{StockhamStage{S}},
                                 xr::Vector{S}, xi::Vector{S}, yr::Vector{S}, yi::Vector{S}) where {S}
    @inbounds for j in 1:n
        o = base + (j - 1) * st - 1
        jb = (j - 1) * B
        for b in 1:B
            z = avin[o+b]
            _sst!(xr, jb + b, real(z))
            _sst!(xi, jb + b, imag(z))
        end
    end
    W = _stockham_width(S)
    Vd = D == -1 ? Val(-1) : Val(1)
    s = B
    ncur = n
    for tw in stages
        R = tw.R
        if R == 4
            _stage4!(yr, yi, xr, xi, ncur, s, tw, Vd, 1, ncur >> 2)
        elseif R == 2
            _stage2!(yr, yi, xr, xi, s, 1, cld(s, W))
        elseif R == 8
            _stage8!(yr, yi, xr, xi, s, Vd, 1, cld(s, W))
        elseif R == 3
            _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(3), D, 1, ncur ÷ 3, 1, cld(s, W))
        elseif R == 5
            _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(5), D, 1, ncur ÷ 5, 1, cld(s, W))
        elseif R == 7
            _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(7), D, 1, ncur ÷ 7, 1, cld(s, W))
        elseif R == 11
            _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(11), D, 1, ncur ÷ 11, 1, cld(s, W))
        else
            _stagep_dir!(yr, yi, xr, xi, ncur, s, tw, Val(13), D, 1, ncur ÷ 13, 1, cld(s, W))
        end
        s *= R
        ncur ÷= R
        xr, xi, yr, yi = yr, yi, xr, xi
    end
    @inbounds for j in 1:n
        o = base + (j - 1) * st - 1
        jb = (j - 1) * B
        for b in 1:B
            avout[o+b] = Complex{S}(xr[jb+b], xi[jb+b])
        end
    end
    return nothing
end
