# FFTA vs FFTW — execution time

FFTA.jl 0.3.1 @ 78e09fd vs FFTW.jl 1.10.0 / FFTW 3.3.11 (fftw), neoverse-n1 (aarch64), Julia 1.12.6, 1 thread, 2026-08-30. Planned execution (`mul!`), minimum over samples; FFTW with its default `ESTIMATE` plans. Produced by `benchmark/summary.jl` from `compare3.jl` output.

## 1 thread

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 6.3 µs | **7.0 µs** | 1.11× |
| fft ComplexF64 2^16 | 1.09 ms | **1.00 ms** | 0.92× |
| fft ComplexF64 2^20 | 52.67 ms | **20.03 ms** | 0.38× |
| fft ComplexF32 2^20 | 24.15 ms | **11.86 ms** | 0.49× |
| fft ComplexF64 1000 (2³·5³) | 10.3 µs | **14.8 µs** | 1.44× |
| fft ComplexF64 10⁶ | 50.42 ms | **52.26 ms** | 1.04× |
| fft ComplexF64 49757 (prime) | 4.57 ms | **4.91 ms** | 1.08× |
| fft ComplexF64 293201 (prime) | 41.02 ms | **45.13 ms** | 1.10× |
| fft ComplexF64 12297 (3·4099) | 777.0 µs | **1.37 ms** | 1.77× |
| rfft Float64 2^12 | 14.5 µs | **19.8 µs** | 1.36× |
| rfft Float64 2^16 | 468.1 µs | **538.4 µs** | 1.15× |
| rfft Float64 2^20 | 13.49 ms | **11.34 ms** | 0.84× |
| rfft Float32 2^16 | 244.0 µs | **306.8 µs** | 1.26× |
| fft ComplexF64 256×256 (2D) | 1.14 ms | **1.20 ms** | 1.05× |
| fft ComplexF64 1024×1024 (2D) | 30.07 ms | **29.84 ms** | 0.99× |
| rfft Float64 1024×1024 (2D) | 8.74 ms | **9.69 ms** | 1.11× |
| fft ComplexF64 64³ (3D) | 2.49 ms | **6.67 ms** | 2.68× |
| fft ComplexF64 4096×64 along dim 1 | 2.16 ms | **2.64 ms** | 1.22× |
| rfft Float64 4096×64 along dim 1 | 1.04 ms | **1.35 ms** | 1.30× |
| fft ComplexF64 64×4096 along dim 2 | 5.35 ms | **6.85 ms** | 1.28× |
| rfft Float64 1024×1024 along dim 1 | 3.67 ms | **4.79 ms** | 1.31× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.90× (20) | 0.93× (20) | 1.20× (20) | 1.23× (20) |
| smooth | 2.12× (24) | 2.63× (24) | 2.59× (24) | 2.76× (24) |
| prime | 1.48× (24) | 1.32× (24) | 1.72× (24) | 1.26× (24) |
| awkward | 1.58× (29) | 1.45× (29) | 1.69× (29) | 1.46× (29) |
| 2d | 1.61× (14) | 1.98× (9) | 1.82× (14) | 2.10× (9) |
| 3d | 2.99× (5) | 4.19× (5) | 3.41× (5) | 4.07× (5) |
| batched, along dim 1 | 1.26× (6) | 1.44× (6) | 1.35× (9) | 1.29× (6) |
| batched, along dim 2 | 1.39× (6) | 1.49× (6) | 1.22× (6) | 1.14× (6) |

**All 505 cases: 1.62× geometric mean; FFTA faster than FFTW in 89 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## 16 threads (FFTW `set_num_threads(16)`, FFTA `num_threads=16`; 1D powers of two, 2D/3D, batched)

FFTW's threaded plans are slower than its serial ones below ~2^17 points (its threading overhead) — the small-transform rows and the `2d` geomean reflect that, not a change in FFTA.

### Representative cases

| case | FFTW | FFTA | FFTA / FFTW |
|:--|--:|--:|--:|
| fft ComplexF64 2^10 | 27.4 µs | **7.0 µs** | 0.25× |
| fft ComplexF64 2^16 | 134.3 µs | **994.1 µs** | 7.40× |
| fft ComplexF64 2^20 | 3.26 ms | **1.96 ms** | 0.60× |
| fft ComplexF32 2^20 | 1.91 ms | **1.12 ms** | 0.59× |
| rfft Float64 2^12 | 14.6 µs | **19.8 µs** | 1.36× |
| rfft Float64 2^16 | 331.2 µs | **534.3 µs** | 1.61× |
| rfft Float64 2^20 | 5.31 ms | **3.46 ms** | 0.65× |
| rfft Float32 2^16 | 224.6 µs | **312.6 µs** | 1.39× |
| fft ComplexF64 256×256 (2D) | 2.87 ms | **163.1 µs** | 0.06× |
| fft ComplexF64 1024×1024 (2D) | 2.46 ms | **3.41 ms** | 1.39× |
| rfft Float64 1024×1024 (2D) | 2.58 ms | **762.4 µs** | 0.30× |
| fft ComplexF64 64³ (3D) | 255.8 µs | **857.5 µs** | 3.35× |
| fft ComplexF64 4096×64 along dim 1 | 195.2 µs | **150.0 µs** | 0.77× |
| rfft Float64 4096×64 along dim 1 | 100.5 µs | **81.2 µs** | 0.81× |
| fft ComplexF64 64×4096 along dim 2 | 430.2 µs | **434.6 µs** | 1.01× |
| rfft Float64 1024×1024 along dim 1 | 301.6 µs | **282.0 µs** | 0.94× |

## By size class (geometric mean of FFTA / FFTW, lower is better)

| class | ComplexF64 fft | ComplexF32 fft | Float64 rfft | Float32 rfft |
|:--|--:|--:|--:|--:|
| pow2 | 0.70× (20) | 0.87× (20) | 0.72× (20) | 0.93× (20) |
| smooth | — | — | — | — |
| prime | — | — | — | — |
| awkward | — | — | — | — |
| 2d | 0.82× (14) | 0.60× (9) | 0.32× (14) | 0.46× (9) |
| 3d | 1.48× (5) | 1.53× (5) | 1.41× (5) | 1.47× (5) |
| batched, along dim 1 | 1.18× (6) | 1.13× (6) | 1.02× (9) | 0.97× (6) |
| batched, along dim 2 | 1.51× (6) | 1.39× (6) | 1.17× (6) | 0.88× (6) |

**All 197 cases: 0.84× geometric mean; FFTA faster than FFTW in 100 of them.** Classes: `pow2` = 2^k; `smooth` = 2^a·3^b·5^c·7^d; `prime`; `awkward` = prime × small factor; (n) = number of cases.


## DSP.jl test suite

Time of the `DSP.jl` testset (from the test summary, excludes compilation) for DSP.jl (`ffta` branch of pankgeorg/DSP.jl = DSP.jl with this FFTA as its FFT provider, vs DSP.jl master on FFTW), 1 thread, same machine, back to back; both runs pass the same 12 011 tests.

| provider | run 1 | run 2 |
|:--|--:|--:|
| FFTW | 438 s | 433 s |
| FFTA | 787 s | 787 s |

The whole difference is `test/dsp.jl` (`conv`/`xcorr`/`deconv`): 224 s on FFTW, 570 s on FFTA; every other testset is within 10 %. `dsp_cases.jl` replays that file's FFT calls and separates the causes (`DSP_CASES.md`): the transforms themselves account for ~1.5 s of it (2-D/3-D convolutions at 7-smooth sizes such as 140 = 2²·5·7, where FFTA's 1-D kernel is 3× FFTW's); the rest is **first-call compilation** — FFTA's generated codelets are specialised on the input array type, and the N-d path passes strided views, so every new (element type, dimensionality) combination recompiles them; the Float32 wide-lane codelets take ~45 s each (2-D and 3-D Float32 real plans: 43 s and 58 s on first use, 0.6 ms after).
