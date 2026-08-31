# FFTA vs FFTW — execution time

FFTA.jl 0.3.1 @ 4f9ff88 vs FFTW.jl 1.10.0 / FFTW 3.3.11 (fftw), neoverse-n1 (aarch64), Julia 1.12.6, 1 thread, 2026-08-30. Planned execution (`mul!`), minimum over samples; FFTW with its default `ESTIMATE` plans. Produced by `benchmark/summary.jl` from `compare3.jl` output.

## 1 thread

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 6.3 µs | **7.0 µs** | 1.11× |
| fft ComplexF64 2^16 | 1.09 ms | **976.9 µs** | 0.90× |
| fft ComplexF64 2^20 | 52.67 ms | **19.29 ms** | 0.37× |
| fft ComplexF32 2^20 | 24.15 ms | **11.17 ms** | 0.46× |
| fft ComplexF64 1000 (2³·5³) | 10.3 µs | **9.4 µs** | 0.91× |
| fft ComplexF64 10⁶ | 50.42 ms | **17.90 ms** | 0.36× |
| fft ComplexF64 49757 (prime) | 4.57 ms | **3.26 ms** | 0.71× |
| fft ComplexF64 293201 (prime) | 41.02 ms | **23.07 ms** | 0.56× |
| fft ComplexF64 12297 (3·4099) | 777.0 µs | **729.0 µs** | 0.94× |
| rfft Float64 2^12 | 14.5 µs | **20.3 µs** | 1.40× |
| rfft Float64 2^16 | 468.1 µs | **545.2 µs** | 1.16× |
| rfft Float64 2^20 | 13.49 ms | **11.35 ms** | 0.84× |
| rfft Float32 2^16 | 244.0 µs | **241.3 µs** | 0.99× |
| fft ComplexF64 256×256 (2D) | 1.14 ms | **1.08 ms** | 0.95× |
| fft ComplexF64 1024×1024 (2D) | 30.07 ms | **26.24 ms** | 0.87× |
| rfft Float64 1024×1024 (2D) | 8.74 ms | **9.99 ms** | 1.14× |
| fft ComplexF64 64³ (3D) | 2.49 ms | **5.92 ms** | 2.38× |
| fft ComplexF64 4096×64 along dim 1 | 2.16 ms | **2.58 ms** | 1.19× |
| rfft Float64 4096×64 along dim 1 | 1.04 ms | **1.34 ms** | 1.29× |
| fft ComplexF64 64×4096 along dim 2 | 5.35 ms | **4.33 ms** | 0.81× |
| rfft Float64 1024×1024 along dim 1 | 3.67 ms | **4.83 ms** | 1.32× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.88× (20) | 0.74× (20) | 1.20× (20) | 1.04× (20) |
| smooth | 1.01× (24) | 0.92× (24) | 1.40× (24) | 1.15× (24) |
| prime | 1.16× (24) | 0.89× (24) | 1.36× (24) | 0.85× (24) |
| awkward | 1.02× (29) | 0.84× (29) | 1.07× (29) | 0.82× (29) |
| 2d | 1.29× (14) | 1.17× (9) | 1.65× (14) | 1.82× (9) |
| 3d | 2.63× (5) | 2.44× (5) | 3.29× (5) | 3.12× (5) |
| batched, along dim 1 | 1.19× (6) | 0.91× (6) | 1.35× (9) | 1.04× (6) |
| batched, along dim 2 | 0.99× (6) | 0.55× (6) | 1.22× (6) | 1.02× (6) |

**All 505 cases: 1.08× geometric mean; FFTA faster than FFTW in 261 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## 16 threads (FFTW `set_num_threads(16)`, FFTA `num_threads=16`)

FFTA threads across pencils, inside single transforms of ≥ 2¹⁷ points (Stockham task team; pow2 ComplexF64 ≥ 2¹⁸ via leaves-first), and through batched N-d pencils.

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 29.1 µs | **7.0 µs** | 0.24× |
| fft ComplexF64 2^16 | 141.9 µs | **977.9 µs** | 6.89× |
| fft ComplexF64 2^20 | 3.04 ms | **1.93 ms** | 0.63× |
| fft ComplexF32 2^20 | 1.85 ms | **1.24 ms** | 0.67× |
| fft ComplexF64 1000 (2³·5³) | 40.2 µs | **9.3 µs** | 0.23× |
| fft ComplexF64 10⁶ | 2.68 ms | **1.71 ms** | 0.64× |
| fft ComplexF64 49757 (prime) | 1.52 ms | **3.35 ms** | 2.20× |
| fft ComplexF64 293201 (prime) | 13.07 ms | **6.20 ms** | 0.47× |
| fft ComplexF64 12297 (3·4099) | 269.4 µs | **727.6 µs** | 2.70× |
| rfft Float64 2^12 | 14.5 µs | **20.3 µs** | 1.40× |
| rfft Float64 2^16 | 338.1 µs | **557.4 µs** | 1.65× |
| rfft Float64 2^20 | 5.38 ms | **3.85 ms** | 0.71× |
| rfft Float32 2^16 | 227.4 µs | **238.5 µs** | 1.05× |
| fft ComplexF64 256×256 (2D) | 2.99 ms | **88.6 µs** | 0.03× |
| fft ComplexF64 1024×1024 (2D) | 2.46 ms | **1.75 ms** | 0.71× |
| rfft Float64 1024×1024 (2D) | 2.65 ms | **741.8 µs** | 0.28× |
| fft ComplexF64 64³ (3D) | 259.6 µs | **489.2 µs** | 1.88× |
| fft ComplexF64 4096×64 along dim 1 | 190.8 µs | **147.2 µs** | 0.77× |
| rfft Float64 4096×64 along dim 1 | 101.8 µs | **84.2 µs** | 0.83× |
| fft ComplexF64 64×4096 along dim 2 | 445.3 µs | **285.0 µs** | 0.64× |
| rfft Float64 1024×1024 along dim 1 | 302.4 µs | **291.4 µs** | 0.96× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.62× (20) | 0.64× (20) | 0.72× (20) | 0.79× (20) |
| smooth | 0.36× (24) | 0.30× (24) | 0.64× (24) | 0.44× (24) |
| prime | 0.40× (24) | 0.22× (24) | 1.24× (24) | 0.76× (24) |
| awkward | 2.21× (29) | 1.67× (29) | 1.49× (29) | 1.06× (29) |
| 2d | 0.54× (14) | 0.36× (9) | 0.29× (14) | 0.40× (9) |
| 3d | 1.03× (5) | 1.04× (5) | 1.38× (5) | 1.19× (5) |
| batched, along dim 1 | 1.13× (6) | 0.75× (6) | 1.02× (9) | 0.79× (6) |
| batched, along dim 2 | 1.12× (6) | 1.55× (6) | 1.11× (6) | 0.80× (6) |

**All 505 cases: 0.72× geometric mean; FFTA faster than FFTW in 266 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## DSP.jl test suite

Time of the `DSP.jl` testset (test summary, excludes compilation), 1 thread, same machine, back to back; both providers pass the same 12 011 tests.

| provider | time |
|:--|--:|
| FFTW | 431 s |
| FFTA | **474 s** |

The conv-heavy `test/dsp.jl` file accounts for the difference; the FFT calls it makes are replayed and dissected in `DSP_CASES.md`.
