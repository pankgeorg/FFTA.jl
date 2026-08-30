#!/usr/bin/env julia
#=
Where does the time go *inside* FFTA's kernels?

For a few representative sizes the planned execution is split into its
stages by re-running the kernel with stages left out (the recursion is
copied here with the butterfly pass or the leaves removed), so that the
stage times add up to the measured total. Each stage is put next to the
time a plain memory pass over the same data takes (a `copyto!` of the
array, measured in the same process), which is the bound a stage cannot
beat, and next to FFTW's ESTIMATE/MEASURE totals.

    julia --project=. -t 1 kernel_stages.jl [--sizes 1024,16384,1048576] [--seconds 0.3]

Reads FFTA from the parent directory (`Pkg.develop`), like `suite.jl`.
=#
import Pkg
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()
using FFTA, FFTW, Statistics, Printf, LinearAlgebra
using FFTA: fft_pow2_radix4!, _pow2_codelet!, CODELET_MAX, pow2_twiddles, direction_sign,
            FFT_FORWARD, CallGraph, fft_kernel!, COMPOSITE_FFT, BLUESTEIN, POW2RADIX4_FFT

function getopt(flag, default)
    i = findfirst(==(flag), ARGS)
    i === nothing ? default : ARGS[i + 1]
end
const SECONDS = parse(Float64, getopt("--seconds", "0.3"))
const SIZES = parse.(Int, split(getopt("--sizes", "1024,16384,262144,4194304,1000,46305,1000000,65537"), ","))
const T = ComplexF64

function timeit(f; seconds = SECONDS)
    f(); f()
    t0 = time_ns(); f(); t1 = time_ns()
    evals = max(1, ceil(Int, 20_000 / max(t1 - t0, 1)))
    best = Inf
    deadline = time_ns() + seconds * 1e9
    while time_ns() < deadline
        ts = time_ns()
        for _ in 1:evals; f(); end
        best = min(best, (time_ns() - ts) / evals)
    end
    return best * 1e-9
end
fmt(t) = t < 1e-6 ? @sprintf("%.0f ns", t * 1e9) : t < 1e-3 ? @sprintf("%.2f µs", t * 1e6) : @sprintf("%.3f ms", t * 1e3)
pct(t, tot) = @sprintf("%.0f%%", 100t / tot)

# ---------------------------------------------------------------------------
# radix-4 recursion with stages removed (mirrors `fft_pow2_radix4!`)
# ---------------------------------------------------------------------------
# leaves only: the recursion down to the codelet size, no butterfly passes
function pow2_leaves!(out, in, N, so, sto, si, sti, d, tw, toff)
    if N <= CODELET_MAX
        _pow2_codelet!(out, in, N, so, sto, si, sti, d) && return
        fft_pow2_radix4!(out, in, N, so, sto, si, sti, d, tw, toff); return
    end
    m = N ÷ 4; tn = toff + 3m
    pow2_leaves!(out, in, m, so,           sto, si,         sti * 4, d, tw, tn)
    pow2_leaves!(out, in, m, so +   m*sto, sto, si +   sti, sti * 4, d, tw, tn)
    pow2_leaves!(out, in, m, so + 2*m*sto, sto, si + 2*sti, sti * 4, d, tw, tn)
    pow2_leaves!(out, in, m, so + 3*m*sto, sto, si + 3*sti, sti * 4, d, tw, tn)
end
# one butterfly pass of the level of size N (data already in `out`)
function pow2_pass!(out, N, so, sto, d, tw, toff)
    m = N ÷ 4
    minusi = -direction_sign(d) * im
    @inbounds for k in 0:m-1
        wkoe = tw[toff + 3k + 1]; wkeo = tw[toff + 3k + 2]; wkoo = tw[toff + 3k + 3]
        kee = so + k * sto; koe = so + (k + m) * sto; keo = so + (k + 2m) * sto; koo = so + (k + 3m) * sto
        y_kee, y_koe, y_keo, y_koo = out[kee], out[koe], out[keo], out[koo]
        t_koe = y_koe * wkoe; t_keo = y_keo * wkeo; t_koo = y_koo * wkoo
        a = y_kee + t_keo; b = y_kee - t_keo
        c = t_koe + t_koo; e = -(t_koe - t_koo) * minusi
        out[kee] = a + c; out[koe] = b + e; out[keo] = a - c; out[koo] = b - e
    end
end
# all butterfly passes of one recursion level (the level whose blocks have size N)
function pow2_level!(out, N, level_size, so, sto, d, tw, toff)
    if N == level_size
        pow2_pass!(out, N, so, sto, d, tw, toff); return
    end
    m = N ÷ 4; tn = toff + 3m
    for q in 0:3
        pow2_level!(out, m, level_size, so + q*m*sto, sto, d, tw, tn)
    end
end
# the same pass with the arithmetic removed: 4 strided read streams, 4 write streams, 3 twiddle reads
function pow2_pass_mem!(out, N, so, sto, d, tw, toff)
    m = N ÷ 4
    @inbounds for k in 0:m-1
        wkoe = tw[toff + 3k + 1]; wkeo = tw[toff + 3k + 2]; wkoo = tw[toff + 3k + 3]
        kee = so + k * sto; koe = so + (k + m) * sto; keo = so + (k + 2m) * sto; koo = so + (k + 3m) * sto
        y_kee, y_koe, y_keo, y_koo = out[kee], out[koe], out[keo], out[koo]
        out[kee] = y_kee + wkoe; out[koe] = y_koe + wkeo; out[keo] = y_keo + wkoo; out[koo] = y_koo
    end
end
function pow2_level_mem!(out, N, level_size, so, sto, d, tw, toff)
    if N == level_size
        pow2_pass_mem!(out, N, so, sto, d, tw, toff); return
    end
    m = N ÷ 4; tn = toff + 3m
    for q in 0:3
        pow2_level_mem!(out, m, level_size, so + q*m*sto, sto, d, tw, tn)
    end
end

function stages_pow2(N, io)
    x = randn(T, N); y = similar(x); y2 = similar(x)
    d = FFT_FORWARD
    tw = pow2_twiddles(T, N, d)
    t_total = timeit(() -> fft_pow2_radix4!(y, x, N, 1, 1, 1, 1, d, tw, 0))
    t_leaves = timeit(() -> pow2_leaves!(y, x, N, 1, 1, 1, 1, d, tw, 0))
    leaf = let l = N
        while l > CODELET_MAX; l ÷= 4; end
        l   # (a fresh binding: a reassigned local would be boxed in the closures below)
    end
    levels = Int[]; M = N
    while M > leaf; push!(levels, M); M ÷= 4; end
    t_copy = timeit(() -> copyto!(y2, y))
    # contiguous codelets of the leaf size on the same data (leaf work without the strided reads)
    nleaf = N ÷ leaf
    t_leaf_contig = timeit(() -> (for b in 0:nleaf-1; fft_pow2_radix4!(y, x, leaf, 1 + b*leaf, 1, 1 + b*leaf, 1, d, tw, 0); end))
    pw_e = plan_fft(x; flags = FFTW.ESTIMATE); pw_m = plan_fft(x; flags = FFTW.MEASURE)
    t_fftw_e = timeit(() -> mul!(y2, pw_e, x)); t_fftw_m = timeit(() -> mul!(y2, pw_m, x))
    println(io, "\n### n = $N = 2^$(round(Int, log2(N))) (radix-4 recursion, $(length(levels)) butterfly levels, $nleaf leaves of $leaf points)\n")
    println(io, "| stage | time | share of FFTA total | time of one `copyto!` of the array: $(fmt(t_copy)) |")
    println(io, "|:--|--:|--:|--:|")
    @printf(io, "| FFTA total | %s | 100%% | %.1f× copy |\n", fmt(t_total), t_total / t_copy)
    @printf(io, "| leaves: %d × %d-point codelet, input stride %d | %s | %s | %.1f× copy |\n", nleaf, leaf, N ÷ leaf, fmt(t_leaves), pct(t_leaves, t_total), t_leaves / t_copy)
    @printf(io, "| (same codelets on contiguous input) | %s | %s | strided-load penalty %.2f× |\n", fmt(t_leaf_contig), pct(t_leaf_contig, t_total), t_leaves / t_leaf_contig)
    sum_levels = 0.0
    for L in levels
        t_lvl = timeit(() -> pow2_level!(y, N, L, 1, 1, d, tw, 0))
        t_mem = timeit(() -> pow2_level_mem!(y, N, L, 1, 1, d, tw, 0))
        sum_levels += t_lvl
        @printf(io, "| butterfly pass, blocks of %d (stride %d between the 4 legs) | %s | %s | %.1f× copy; memory-only version of the pass %s |\n",
                L, L ÷ 4, fmt(t_lvl), pct(t_lvl, t_total), t_lvl / t_copy, fmt(t_mem))
    end
    @printf(io, "| leaves + passes (sum of the rows) | %s | %s | recursion/dispatch overhead = total − sum = %s |\n", fmt(t_leaves + sum_levels), pct(t_leaves + sum_levels, t_total), fmt(t_total - t_leaves - sum_levels))
    @printf(io, "| FFTW ESTIMATE / MEASURE | %s / %s | %s / %s | %.1f× / %.1f× copy |\n", fmt(t_fftw_e), fmt(t_fftw_m), pct(t_fftw_e, t_total), pct(t_fftw_m, t_total), t_fftw_e / t_copy, t_fftw_m / t_copy)
end

# ---------------------------------------------------------------------------
# composite N = N1·N2 (mirrors `fft_composite!`)
# ---------------------------------------------------------------------------
function composite_stages!(out, in, g, idx, d; right = true, twiddle = true, left = true)
    root = g[idx]; li = idx + root.left; ri = idx + root.right
    N1 = g[li].sz; N2 = g[ri].sz
    tmp = g.workspace[idx]; tw = g.twiddles[idx]
    for j1 in 0:N1-1
        right && fft_kernel!(tmp, in, 1 + N2*j1, 1 + j1*root.s_in, d, g[ri].type, g, ri)
        if twiddle && j1 > 0
            base = (j1 - 1) * (N2 - 1); R = 1 + N2*j1
            @inbounds for k2 in 1:N2-1
                tmp[R + k2] *= tw[base + k2]
            end
        end
    end
    left && for k2 in 0:N2-1
        fft_kernel!(out, tmp, 1 + k2*root.s_out, 1 + k2, d, g[li].type, g, li)
    end
end
function stages_composite(N, io)
    x = randn(T, N); y = similar(x); y2 = similar(x)
    d = FFT_FORWARD
    g = CallGraph{T}(N, FFTA.DEFAULT_BLUESTEIN_CUTOFF, d)
    g[1].type === COMPOSITE_FFT || return
    root = g[1]; N1 = g[1 + root.left].sz; N2 = g[1 + root.right].sz
    t_total = timeit(() -> fft_kernel!(y, x, 1, 1, d, g[1].type, g, 1))
    t_right = timeit(() -> composite_stages!(y, x, g, 1, d; twiddle = false, left = false))
    t_tw    = timeit(() -> composite_stages!(y, x, g, 1, d; right = false, left = false))
    t_left  = timeit(() -> composite_stages!(y, x, g, 1, d; right = false, twiddle = false))
    t_copy = timeit(() -> copyto!(y2, y))
    pw_e = plan_fft(x; flags = FFTW.ESTIMATE); pw_m = plan_fft(x; flags = FFTW.MEASURE)
    t_fftw_e = timeit(() -> mul!(y2, pw_e, x)); t_fftw_m = timeit(() -> mul!(y2, pw_m, x))
    desc(i) = string(g[i].sz, " ", g[i].type)
    println(io, "\n### n = $N = $N1 × $N2 (composite step: right = $(desc(1 + root.right)), left = $(desc(1 + root.left)))\n")
    println(io, "| stage | time | share of FFTA total | time of one `copyto!` of the array: $(fmt(t_copy)) |")
    println(io, "|:--|--:|--:|--:|")
    @printf(io, "| FFTA total | %s | 100%% | %.1f× copy |\n", fmt(t_total), t_total / t_copy)
    @printf(io, "| %d right sub-transforms of %d (input stride %d) | %s | %s | %.1f× copy |\n", N1, N2, root.s_in * N1, fmt(t_right), pct(t_right, t_total), t_right / t_copy)
    @printf(io, "| twiddle multiply pass | %s | %s | %.1f× copy |\n", fmt(t_tw), pct(t_tw, t_total), t_tw / t_copy)
    @printf(io, "| %d left sub-transforms of %d (output stride %d) | %s | %s | %.1f× copy |\n", N2, N1, N2 * root.s_out, fmt(t_left), pct(t_left, t_total), t_left / t_copy)
    @printf(io, "| sum of the rows | %s | %s | overhead = %s |\n", fmt(t_right + t_tw + t_left), pct(t_right + t_tw + t_left, t_total), fmt(t_total - t_right - t_tw - t_left))
    @printf(io, "| FFTW ESTIMATE / MEASURE | %s / %s | %s / %s | %.1f× / %.1f× copy |\n", fmt(t_fftw_e), fmt(t_fftw_m), pct(t_fftw_e, t_total), pct(t_fftw_m, t_total), t_fftw_e / t_copy, t_fftw_m / t_copy)
end

# ---------------------------------------------------------------------------
# Bluestein (mirrors `fft_bluestein!`)
# ---------------------------------------------------------------------------
function blue_fill!(a, x, chirp, N, pad_len)
    @inbounds for i in 1:N
        a[i] = x[i] * conj(chirp[i])
    end
    @inbounds for i in N+1:pad_len
        a[i] = zero(eltype(a))
    end
end
function blue_pointwise!(tmp, chirp_fft, pad_len)
    @inbounds for i in 1:pad_len
        tmp[i] = conj(tmp[i] * chirp_fft[i])
    end
end
function blue_out!(y, a, chirp, N)
    @inbounds for i in 1:N
        y[i] = conj(chirp[i]) * conj(a[i])
    end
end
function stages_bluestein(N, io)
    x = randn(T, N); y = similar(x); y2 = similar(x)
    d = FFT_FORWARD
    g = CallGraph{T}(N, FFTA.DEFAULT_BLUESTEIN_CUTOFF, d)
    g[1].type === BLUESTEIN || return
    sc = g.bluestein[1]
    (; pad_len, chirp, chirp_fft, a, tmp, graph) = sc
    gt = graph[1].type
    t_total = timeit(() -> fft_kernel!(y, x, 1, 1, d, g[1].type, g, 1))
    t_fill = timeit(() -> blue_fill!(a, x, chirp, N, pad_len))
    t_fft1 = timeit(() -> fft_kernel!(tmp, a, 1, 1, FFT_FORWARD, gt, graph, 1))
    t_pw   = timeit(() -> blue_pointwise!(tmp, chirp_fft, pad_len))
    t_fft2 = timeit(() -> fft_kernel!(a, tmp, 1, 1, FFT_FORWARD, gt, graph, 1))
    t_out  = timeit(() -> blue_out!(y, a, chirp, N))
    t_copy = timeit(() -> copyto!(y2, y))
    pw_e = plan_fft(x; flags = FFTW.ESTIMATE); pw_m = plan_fft(x; flags = FFTW.MEASURE)
    t_fftw_e = timeit(() -> mul!(y2, pw_e, x)); t_fftw_m = timeit(() -> mul!(y2, pw_m, x))
    println(io, "\n### n = $N (Bluestein, padded to $pad_len)\n")
    println(io, "| stage | time | share of FFTA total | time of one `copyto!` of the array: $(fmt(t_copy)) |")
    println(io, "|:--|--:|--:|--:|")
    @printf(io, "| FFTA total | %s | 100%% | %.1f× copy |\n", fmt(t_total), t_total / t_copy)
    @printf(io, "| chirp multiply + zero pad | %s | %s | |\n", fmt(t_fill), pct(t_fill, t_total))
    @printf(io, "| forward FFT of %d | %s | %s | |\n", pad_len, fmt(t_fft1), pct(t_fft1, t_total))
    @printf(io, "| pointwise product | %s | %s | |\n", fmt(t_pw), pct(t_pw, t_total))
    @printf(io, "| second FFT of %d | %s | %s | |\n", pad_len, fmt(t_fft2), pct(t_fft2, t_total))
    @printf(io, "| output chirp multiply | %s | %s | |\n", fmt(t_out), pct(t_out, t_total))
    @printf(io, "| FFTW ESTIMATE / MEASURE | %s / %s | %s / %s | |\n", fmt(t_fftw_e), fmt(t_fftw_m), pct(t_fftw_e, t_total), pct(t_fftw_m, t_total))
end

io = stdout
println(io, "# Kernel stage breakdown (ComplexF64, single thread, $(Sys.CPU_NAME), Julia $(VERSION))")
println(io, "\nStage times are minimum times of the kernel re-run with the other stages removed; `copy` is one `copyto!` of the array (the memory-traffic floor of a single pass).")
for N in SIZES
    if ispow2(N)
        stages_pow2(N, io)
    elseif FFTA.Primes.isprime(N)
        stages_bluestein(N, io)
    else
        stages_composite(N, io)
    end
    flush(io)
end
