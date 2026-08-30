# FFTA vs FFTW — execution time

FFTA.jl 0.3.1 @ a2fd368 vs FFTW.jl 1.10.0 / FFTW 3.3.11 (fftw), neoverse-n1 (aarch64), Julia 1.12.6, 1 thread, 2026-08-30. Planned execution (`mul!`), minimum over samples; FFTW with its default `ESTIMATE` plans. Produced by `benchmark/summary.jl` from `compare3.jl` output.

## 1 thread

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 6.3 µs | **7.0 µs** | 1.10× |
| fft ComplexF64 2^16 | 1.09 ms | **992.3 µs** | 0.91× |
| fft ComplexF64 2^20 | 52.67 ms | **19.40 ms** | 0.37× |
| fft ComplexF32 2^20 | 24.15 ms | **11.22 ms** | 0.46× |
| fft ComplexF64 1000 (2³·5³) | 10.3 µs | **9.2 µs** | 0.89× |
| fft ComplexF64 10⁶ | 50.42 ms | **17.76 ms** | 0.35× |
| fft ComplexF64 49757 (prime) | 4.57 ms | **3.27 ms** | 0.72× |
| fft ComplexF64 293201 (prime) | 41.02 ms | **22.97 ms** | 0.56× |
| fft ComplexF64 12297 (3·4099) | 777.0 µs | **721.2 µs** | 0.93× |
| rfft Float64 2^12 | 14.5 µs | **20.3 µs** | 1.40× |
| rfft Float64 2^16 | 468.1 µs | **548.2 µs** | 1.17× |
| rfft Float64 2^20 | 13.49 ms | **12.22 ms** | 0.91× |
| rfft Float32 2^16 | 244.0 µs | **242.3 µs** | 0.99× |
| fft ComplexF64 256×256 (2D) | 1.14 ms | **1.20 ms** | 1.05× |
| fft ComplexF64 1024×1024 (2D) | 30.07 ms | **26.87 ms** | 0.89× |
| rfft Float64 1024×1024 (2D) | 8.74 ms | **9.83 ms** | 1.12× |
| fft ComplexF64 64³ (3D) | 2.49 ms | **6.57 ms** | 2.64× |
| fft ComplexF64 4096×64 along dim 1 | 2.16 ms | **2.53 ms** | 1.17× |
| rfft Float64 4096×64 along dim 1 | 1.04 ms | **1.34 ms** | 1.29× |
| fft ComplexF64 64×4096 along dim 2 | 5.35 ms | **4.30 ms** | 0.80× |
| rfft Float64 1024×1024 along dim 1 | 3.67 ms | **4.78 ms** | 1.30× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.89× (20) | 0.72× (20) | 1.22× (20) | 1.03× (20) |
| smooth | 0.98× (24) | 0.90× (24) | 1.38× (24) | 1.14× (24) |
| prime | 1.15× (24) | 0.87× (24) | 1.34× (24) | 0.82× (24) |
| awkward | 1.02× (29) | 0.83× (29) | 1.07× (29) | 0.81× (29) |
| 2d | 1.41× (14) | 1.62× (9) | 1.64× (14) | 1.75× (9) |
| 3d | 2.88× (5) | 3.36× (5) | 3.30× (5) | 3.25× (5) |
| batched, along dim 1 | 1.22× (6) | 1.01× (6) | 1.34× (9) | 1.01× (6) |
| batched, along dim 2 | 0.97× (6) | 0.87× (6) | 1.19× (6) | 1.00× (6) |

**All 505 cases: 1.09× geometric mean; FFTA faster than FFTW in 257 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## 16 threads (FFTW `set_num_threads(16)`, FFTA `num_threads=16`)

FFTA threads across pencils and inside pow2 transforms ≥ 2^18; other single 1-D transforms are serial (threading the Stockham stages is in progress). FFTW's threaded plans are slower than serial below ~2^17 points.

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 29.1 µs | **7.0 µs** | 0.24× |
| fft ComplexF64 2^16 | 141.9 µs | **1.03 ms** | 7.23× |
| fft ComplexF64 2^20 | 3.04 ms | **1.82 ms** | 0.60× |
| fft ComplexF32 2^20 | 1.85 ms | **1.12 ms** | 0.61× |
| fft ComplexF64 1000 (2³·5³) | 40.2 µs | **9.1 µs** | 0.23× |
| fft ComplexF64 10⁶ | 2.68 ms | **17.88 ms** | 6.66× |
| fft ComplexF64 49757 (prime) | 1.52 ms | **3.21 ms** | 2.11× |
| fft ComplexF64 293201 (prime) | 13.07 ms | **23.87 ms** | 1.83× |
| fft ComplexF64 12297 (3·4099) | 269.4 µs | **730.6 µs** | 2.71× |
| rfft Float64 2^12 | 14.5 µs | **20.3 µs** | 1.40× |
| rfft Float64 2^16 | 338.1 µs | **591.9 µs** | 1.75× |
| rfft Float64 2^20 | 5.38 ms | **3.68 ms** | 0.68× |
| rfft Float32 2^16 | 227.4 µs | **238.6 µs** | 1.05× |
| fft ComplexF64 256×256 (2D) | 2.99 ms | **158.5 µs** | 0.05× |
| fft ComplexF64 1024×1024 (2D) | 2.46 ms | **3.22 ms** | 1.31× |
| rfft Float64 1024×1024 (2D) | 2.65 ms | **757.4 µs** | 0.29× |
| fft ComplexF64 64³ (3D) | 259.6 µs | **824.0 µs** | 3.17× |
| fft ComplexF64 4096×64 along dim 1 | 190.8 µs | **157.2 µs** | 0.82× |
| rfft Float64 4096×64 along dim 1 | 101.8 µs | **83.2 µs** | 0.82× |
| fft ComplexF64 64×4096 along dim 2 | 445.3 µs | **281.9 µs** | 0.63× |
| rfft Float64 1024×1024 along dim 1 | 302.4 µs | **291.2 µs** | 0.96× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.68× (20) | 0.68× (20) | 0.72× (20) | 0.77× (20) |
| smooth | 0.62× (24) | 0.51× (24) | 0.63× (24) | 0.43× (24) |
| prime | 0.47× (24) | 0.25× (24) | 1.22× (24) | 0.74× (24) |
| awkward | 2.84× (29) | 2.03× (29) | 1.47× (29) | 1.06× (29) |
| 2d | 0.75× (14) | 0.50× (9) | 0.29× (14) | 0.39× (9) |
| 3d | 1.42× (5) | 1.28× (5) | 1.38× (5) | 1.18× (5) |
| batched, along dim 1 | 1.22× (6) | 0.83× (6) | 1.01× (9) | 0.78× (6) |
| batched, along dim 2 | 1.11× (6) | 0.73× (6) | 1.10× (6) | 0.79× (6) |

**All 505 cases: 0.80× geometric mean; FFTA faster than FFTW in 248 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## DSP.jl test suite

Time of the `DSP.jl` testset (test summary, excludes compilation), 1 thread, same machine, back to back; both providers pass the same 12 011 tests.

| provider | time |
|:--|--:|
| FFTW | 431 s |
| FFTA | **474 s** |

The conv-heavy `test/dsp.jl` file accounts for the difference; the FFT calls it makes are replayed and dissected in `DSP_CASES.md`.
