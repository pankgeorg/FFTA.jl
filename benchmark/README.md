# Benchmarks

`RESULTS.md` is the comparison that matters: execution time of FFTA (this
branch) against FFTW.jl on the same cases, one machine, one session. It is
rendered by `summary.jl` from `compare3.jl` output:

```bash
cd benchmark
julia --project=. compare3.jl --impl fftw=fftw --impl ffta=..            # ~1 h at 1 thread
julia --project=. summary.jl compare3_results                             # writes RESULTS.md
julia --project=. compare3.jl --impl fftw=fftw --impl ffta=.. --threads 16 --only 1d,nd,batched --classes pow2
julia --project=. summary.jl compare3_results --threads 16 --out RESULTS_16threads.md
```

`compare3.jl` runs each implementation in its own process and environment
(`--impl NAME=SPEC`, `SPEC` = `fftw`, `@version` for a registry FFTA, or a
checkout path), on the case list in `cases.jl` (1D powers of two, smooth
composites, primes, prime×small; 2D/3D; batched along a dimension; complex
and real, Float64 and Float32), and writes one JSON per column plus a full
per-case table (`COMPARE3.md`). Options: `--threads`, `--only 1d,nd,batched`,
`--classes`, `--kinds fft,rfft`, `--maxlog2`, `--seconds`, `--skip-existing`
(reuse a finished column whose case selection covers the run).

Other files: `kernel_stages.jl` / `KERNEL_STAGES.md` — where the time goes
inside the kernels (stage breakdown); `ANALYSIS.md` — design analysis of the
implementation gaps and structural limits; `suite.jl`/`report.jl` — the
earlier single-process sweep (FFTA reached with `invoke`); `run_benchmarks.jl`
— the original 1D sweep used by the documentation build.
