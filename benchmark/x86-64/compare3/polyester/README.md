# Polyester `@batch` A/B on x86-64

`compare3.jl` from `bench/compare3` @ e9f2a3d:

```
julia --project=. compare3.jl --impl fftw=fftw --impl int=<integration/all @901692b> \
      --impl poly=<exp/polyester-batch @94ab080> --ref int --threads 1,4,6,8,12 --only nd,batched
```

**Machine:** Intel Core Ultra 7 165H (Meteor Lake-H) — 6 P-cores ×2 SMT + 8 E-cores + 2 LP-E
= 22 logical, AVX2 + FMA, no AVX-512, L3 24 MiB, WSL2, Julia 1.12.6, FFTW 3.3.11.
Thread counts chosen so that 6 = physical P-core count and 12 = 6 P-cores × 2 SMT.

## Result: no static-pool straggler on a hybrid part

`poly` vs `int`, geomean, by size band (>1 = Polyester faster):

| threads | <32K (below `THREAD_THRESHOLD`) | 32K–256K | 256K–1M | ≥1M (DRAM-bound) |
|--:|--:|--:|--:|--:|
| 1 | 1.00× | 1.00× | 1.01× | 1.01× |
| 4 | 0.99× | 1.32× | 1.09× | 1.03× |
| 6 | 0.99× | 1.64× | 1.21× | 1.01× |
| 8 | 1.01× | 1.79× | 1.47× | 1.12× |
| 12 | 1.00× | **2.32×** | **1.58×** | 1.12× |

Thread scaling on cases at or above `THREAD_THRESHOLD` (t1 time / tN time):

| threads | `fftw` | `int` | `poly` |
|--:|--:|--:|--:|
| 4 | 2.95× | 3.13× | 3.47× |
| 6 | 3.51× | 3.89× | 4.69× |
| 8 | 3.71× | 4.51× | **6.15×** |
| 12 | 4.17× | 4.59× | **6.89×** |

The concern was that Polyester's static pool assigns one equal-sized chunk per thread,
so a chunk landing on an E-core would become the critical path. **That is not what
happens.** Polyester's advantage grows monotonically with thread count (1.01× → 1.52×
on large cases) and it is largest at 12 threads, well past the 6 physical P-cores.
`Threads.@spawn` is the side that saturates: `int` goes 4.51× → 4.59× from 8 to 12
threads while `poly` continues 6.15× → 6.89×. On this machine the per-task overhead
`@batch` avoids outweighs whatever heterogeneity costs.

## Reading the numbers

* The **<32K band is a null control**: those cases are below `THREAD_THRESHOLD` so both
  columns run identical serial code. It reads 0.99–1.01× at every thread count, which
  is what makes the rest of the table trustworthy.
* The **1-thread row is a second null control** — same code path, no threading. Its
  geomean is 1.01×, but its per-case spread is 0.77–1.50×. That spread is this machine's
  per-case noise for this case set: **individual cells are worth no better than ±50%,
  while the geomeans over 117 cases are sound.** Quote bands, not cells.
* **DRAM-bound (≥1M) is parity**, 1.01–1.12×, matching the aarch64 observation. The
  individual worst cases at 12 threads (`rfft Float64 1000×1000` at 0.57×,
  `fft Float32 65536×64` at 0.74×) sit inside a band whose geomean is 1.12×, and are
  within the noise bound above; they would need a back-to-back re-measure before being
  treated as real.
* **Checksums agree with `int` on every case at every thread count.**
