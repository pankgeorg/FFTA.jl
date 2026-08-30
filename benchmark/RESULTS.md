# FFTA vs FFTW — execution time

FFTA.jl 0.3.1 @ 4943d2a vs FFTW.jl 1.10.0 / FFTW 3.3.11 (fftw), neoverse-n1 (aarch64), Julia 1.12.6, 1 thread, 2026-08-30. Planned execution (`mul!`), minimum over samples; FFTW with its default `ESTIMATE` plans. Produced by `benchmark/summary.jl` from `compare3.jl` output.

## 1 thread

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 6.3 µs | **7.0 µs** | 1.11× |
| fft ComplexF64 2^16 | 1.09 ms | **985.1 µs** | 0.91× |
| fft ComplexF64 2^20 | 52.67 ms | **20.14 ms** | 0.38× |
| fft ComplexF32 2^20 | 24.15 ms | **11.42 ms** | 0.47× |
| fft ComplexF64 1000 (2³·5³) | 10.3 µs | **14.8 µs** | 1.44× |
| fft ComplexF64 10⁶ | 50.42 ms | **50.50 ms** | 1.00× |
| fft ComplexF64 49757 (prime) | 4.57 ms | **4.98 ms** | 1.09× |
| fft ComplexF64 293201 (prime) | 41.02 ms | **46.54 ms** | 1.13× |
| fft ComplexF64 12297 (3·4099) | 777.0 µs | **1.29 ms** | 1.66× |
| rfft Float64 2^12 | 14.5 µs | **20.3 µs** | 1.40× |
| rfft Float64 2^16 | 468.1 µs | **557.9 µs** | 1.19× |
| rfft Float64 2^20 | 13.49 ms | **11.79 ms** | 0.87× |
| rfft Float32 2^16 | 244.0 µs | **310.9 µs** | 1.27× |
| fft ComplexF64 256×256 (2D) | 1.14 ms | **1.20 ms** | 1.06× |
| fft ComplexF64 1024×1024 (2D) | 30.07 ms | **27.23 ms** | 0.91× |
| rfft Float64 1024×1024 (2D) | 8.74 ms | **9.94 ms** | 1.14× |
| fft ComplexF64 64³ (3D) | 2.49 ms | **6.55 ms** | 2.63× |
| fft ComplexF64 4096×64 along dim 1 | 2.16 ms | **2.52 ms** | 1.17× |
| rfft Float64 4096×64 along dim 1 | 1.04 ms | **1.34 ms** | 1.29× |
| fft ComplexF64 64×4096 along dim 2 | 5.35 ms | **4.31 ms** | 0.81× |
| rfft Float64 1024×1024 along dim 1 | 3.67 ms | **4.79 ms** | 1.31× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.90× (20) | 0.93× (20) | 1.21× (20) | 1.23× (20) |
| smooth | 2.10× (24) | 2.62× (24) | 2.49× (24) | 2.63× (24) |
| prime | 1.48× (24) | 1.31× (24) | 1.72× (24) | 1.26× (24) |
| awkward | 1.57× (29) | 1.44× (29) | 1.66× (29) | 1.41× (29) |
| 2d | 1.59× (14) | 1.98× (9) | 1.82× (14) | 2.11× (9) |
| 3d | 2.86× (5) | 4.01× (5) | 3.29× (5) | 3.84× (5) |
| batched, along dim 1 | 1.22× (6) | 1.45× (6) | 1.34× (9) | 1.30× (6) |
| batched, along dim 2 | 0.98× (6) | 1.10× (6) | 1.20× (6) | 1.14× (6) |

**All 505 cases: 1.60× geometric mean; FFTA faster than FFTW in 95 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## 16 threads (FFTW `set_num_threads(16)`, FFTA `num_threads=16`; 1D powers of two, 2D/3D, batched)

FFTW's threaded plans are slower than its serial ones below ~2^17 points (its threading overhead) — the small-transform rows and the `2d` geomean reflect that, not a change in FFTA.

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 27.4 µs | **7.0 µs** | 0.25× |
| fft ComplexF64 2^16 | 134.3 µs | **999.0 µs** | 7.44× |
| fft ComplexF64 2^20 | 3.26 ms | **1.89 ms** | 0.58× |
| fft ComplexF32 2^20 | 1.91 ms | **1.16 ms** | 0.61× |
| rfft Float64 2^12 | 14.6 µs | **20.4 µs** | 1.40× |
| rfft Float64 2^16 | 331.2 µs | **575.2 µs** | 1.74× |
| rfft Float64 2^20 | 5.31 ms | **3.55 ms** | 0.67× |
| rfft Float32 2^16 | 224.6 µs | **321.3 µs** | 1.43× |
| fft ComplexF64 256×256 (2D) | 2.87 ms | **157.2 µs** | 0.05× |
| fft ComplexF64 1024×1024 (2D) | 2.46 ms | **3.23 ms** | 1.31× |
| rfft Float64 1024×1024 (2D) | 2.58 ms | **744.2 µs** | 0.29× |
| fft ComplexF64 64³ (3D) | 255.8 µs | **830.9 µs** | 3.25× |
| fft ComplexF64 4096×64 along dim 1 | 195.2 µs | **157.6 µs** | 0.81× |
| rfft Float64 4096×64 along dim 1 | 100.5 µs | **83.4 µs** | 0.83× |
| fft ComplexF64 64×4096 along dim 2 | 430.2 µs | **295.8 µs** | 0.69× |
| rfft Float64 1024×1024 along dim 1 | 301.6 µs | **294.0 µs** | 0.97× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.69× (20) | 0.87× (20) | 0.73× (20) | 0.94× (20) |
| smooth | — | — | — | — |
| prime | — | — | — | — |
| awkward | — | — | — | — |
| 2d | 0.81× (14) | 0.60× (9) | 0.32× (14) | 0.47× (9) |
| 3d | 1.40× (5) | 1.46× (5) | 1.35× (5) | 1.40× (5) |
| batched, along dim 1 | 1.25× (6) | 1.17× (6) | 1.05× (9) | 1.00× (6) |
| batched, along dim 2 | 1.13× (6) | 0.87× (6) | 1.09× (6) | 0.89× (6) |

**All 197 cases: 0.82× geometric mean; FFTA faster than FFTW in 105 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## DSP.jl test suite

Time of the `DSP.jl` testset (from the test summary, excludes compilation) for DSP.jl (`ffta` branch of pankgeorg/DSP.jl = DSP.jl with this FFTA as its FFT provider, vs DSP.jl master on FFTW), 1 thread, same machine, back to back; both runs pass the same 12 011 tests.

| provider | run 1 | run 2 |
|:--|--:|--:|
| FFTW | 431 s | — |
| FFTA | 488 s | 485 s |

`benchmark/DSP_CASES.md` replays that suite's convolution FFT calls: steady-state execution 1.05× of FFTW over the whole replay (2-D/3-D at 7-smooth sizes 2–2.5×, small one-shot cases faster than FFTW) and first-call compilation 32 s vs 27 s (was 110 s before the kernels were made to see only `Vector` arguments).
