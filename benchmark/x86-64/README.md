# x86-64 results for the comprehensive benchmark suite

Companion to the aarch64 run in `benchmark/REPORT.md`. Same `suite.jl` and
`report.jl` (from the bench branch @ 73b190c), same knobs
(`--seconds 0.5`, `--maxlog2 22`, `-t 8`), same FFTA baseline
(0.3.1, source byte-identical to `main` @ 7aeb327).

| file | contents |
|:--|:--|
| `REPORT.md` | full rendered report, 520 measured cases |
| `results_suite.json` | raw `suite.jl` output for the baseline run |
| `plots/` | SVG ratio and timing plots |
| `PROVENANCE.md` | which commit each run measured |
| `calib_bluestein_x86.md` | 3-smooth pad cost factor, refitted on x86-64 |
| `calib_cutoff_x86.md` | DFT-vs-Bluestein crossover, refitted on x86-64 |
| `calib_padchoice_x86.md` | pad choice measured end-to-end, A vs E |
| `calib_*.jl` | the scripts that produced the three files above |
| `branches/` | per-branch `suite.jl` JSONs and before/after tables |

## Machine

Intel Core Ultra 7 165H (Meteor Lake-H), 11 physical / 22 logical cores,
AVX2 + FMA, **no AVX-512** (LLVM target `alderlake`), L3 24 MiB, 31 GiB RAM,
Linux under WSL2, Julia 1.12.6, FFTW 3.3.11 with provider `fftw` (not MKL).

Because this part has no AVX-512, FFTW holds 2 `ComplexF64` per vector register
rather than 4. The FFTA/FFTW ratios here are therefore a **lower bound** on the
x86-64 gap, not a typical value for server parts.

Caveat: a laptop part under WSL2 with hybrid P/E cores. Single-threaded rows use
minimum-over-samples and are stable; the 8-thread section is indicative only,
since the scheduler may place a thread on an E-core and sustained multi-threaded
runs can thermally throttle.
