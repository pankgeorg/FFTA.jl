# DSP.jl's convolution workload on FFTA vs FFTW

`dsp_cases.jl` replays the FFT call sequences of DSP.jl's `conv` kernels (one-shot, planning every time, as DSP does) over the size and element-type patterns of `test/dsp.jl`, in one process per implementation; `nd_stages.jl` times an N-d transform one dimension at a time. aarch64 Neoverse-N1, 1 thread, FFTA `main`, `--quick` run (the 256³ references excluded).

**Reading:** steady-state execution of the whole replay is 2.5 s on FFTA vs 1.0 s on FFTW — the 2-D/3-D convolutions at 7-smooth sizes (140 = 2²·5·7: FFTA's 1-D kernel is 3× FFTW's there, plus a 1.5–1.9× pencil overhead along dims 2–3), while the small one-shot cases are *faster* than FFTW. What makes DSP.jl's test file 2.5× slower is the **first-call column: 110 s vs 1.2 s** — compilation. FFTA's generated codelets are specialised on the input array type and the N-d path hands them strided views, so each new (element type, dimensionality) recompiles them; the Float32 wide-lane (W = 4) lockstep codelets cost ~45 s per specialisation (2-D and 3-D Float32 real plans: 43 s and 58 s on first use, 0.6 ms afterwards). Fix direction: give the codelets contiguous `Vector` input (copy strided pencils into a worker buffer) so only the precompiled specialisations exist, and/or shrink the Float32 lockstep codelets.

| group | case | FFTW steady | FFTA steady | FFTA / FFTW | FFTA plans | FFTW first call | FFTA first call |
|:--|:--|--:|--:|--:|--:|--:|--:|
| conv-1D simple | Float64 10⋆10 → n=20 | 10.6 µs | **3.4 µs** | 0.32× | 68% | 61.6 µs | **51.30 ms** |
| conv-1D overlapsave | Float64 10⋆10 → nfft=20, 2 blocks | 10.7 µs | **3.6 µs** | 0.33× | 55% | 42.1 µs | **27.2 µs** |
| conv-1D simple | ComplexF64 10⋆10 → n=20 | 4.8 µs | **4.2 µs** | 0.89× | 76% | 553.2 µs | **118.25 ms** |
| conv-1D overlapsave | ComplexF64 10⋆10 → nfft=20, 2 blocks | 4.9 µs | **4.8 µs** | 0.98× | 68% | 24.7 µs | **38.3 µs** |
| conv-1D simple | Float64 10⋆200 → n=210 | 37.4 µs | **21.3 µs** | 0.57× | 61% | 239.1 µs | **71.2 µs** |
| conv-1D overlapsave | Float64 10⋆200 → nfft=64, 4 blocks | 12.0 µs | **4.5 µs** | 0.38× | 37% | 183.5 µs | **17.9 µs** |
| conv-1D simple | ComplexF64 10⋆200 → n=210 | 43.8 µs | **43.0 µs** | 0.98× | 55% | 290.8 µs | **76.6 µs** |
| conv-1D overlapsave | ComplexF64 10⋆200 → nfft=64, 4 blocks | 23.2 µs | **6.9 µs** | 0.30× | 36% | 201.0 µs | **20.3 µs** |
| conv-1D simple | Float64 200⋆10 → n=210 | 37.7 µs | **21.8 µs** | 0.58× | 58% | 56.9 µs | **47.6 µs** |
| conv-1D overlapsave | Float64 200⋆10 → nfft=64, 4 blocks | 12.1 µs | **4.5 µs** | 0.37× | 37% | 24.9 µs | **13.4 µs** |
| conv-1D simple | ComplexF64 200⋆10 → n=210 | 43.8 µs | **43.0 µs** | 0.98× | 54% | 86.6 µs | **73.1 µs** |
| conv-1D overlapsave | ComplexF64 200⋆10 → nfft=64, 4 blocks | 23.6 µs | **6.8 µs** | 0.29× | 37% | 40.4 µs | **20.2 µs** |
| conv-1D simple | Float64 200⋆200 → n=400 | 44.9 µs | **23.4 µs** | 0.52× | 47% | 1.04 ms | **45.6 µs** |
| conv-1D overlapsave | Float64 200⋆200 → nfft=400, 2 blocks | 46.4 µs | **28.0 µs** | 0.60× | 38% | 62.6 µs | **40.6 µs** |
| conv-1D simple | ComplexF64 200⋆200 → n=400 | 29.0 µs | **25.4 µs** | 0.88× | 36% | 283.7 µs | **48.9 µs** |
| conv-1D overlapsave | ComplexF64 200⋆200 → nfft=400, 2 blocks | 34.1 µs | **36.7 µs** | 1.08× | 25% | 49.0 µs | **49.1 µs** |
| conv-2D simple | Float64 10×20 ⋆ 20×10 → 30×30 | 71.9 µs | **87.4 µs** | 1.22× | 18% | 416.9 µs | **869.64 ms** |
| conv-2D overlapsave | Float64 10×20 ⋆ 20×10 → nfft=40, 2 blocks | 103.2 µs | **147.6 µs** | 1.43× | 9% | 1.39 ms | **232.8 µs** |
| conv-2D simple | ComplexF64 10×20 ⋆ 20×10 → 30×30 | 90.5 µs | **150.2 µs** | 1.66× | 6% | 919.1 µs | **161.68 ms** |
| conv-2D overlapsave | ComplexF64 10×20 ⋆ 20×10 → nfft=40, 2 blocks | 163.0 µs | **220.8 µs** | 1.35× | 2% | 1.20 ms | **283.4 µs** |
| conv-2D simple | Float64 10×20 ⋆ 210×200 → 224×224 | 1.22 ms | **2.43 ms** | 2.00× | 1% | 3.41 ms | **2.88 ms** |
| conv-2D overlapsave | Float64 10×20 ⋆ 210×200 → nfft=128, 6 blocks | 932.8 µs | **1.82 ms** | 1.95× | 1% | 1.03 ms | **1.89 ms** |
| conv-2D simple | ComplexF64 10×20 ⋆ 210×200 → 224×224 | 2.48 ms | **4.77 ms** | 1.92× | 0% | 4.04 ms | **5.45 ms** |
| conv-2D overlapsave | ComplexF64 10×20 ⋆ 210×200 → nfft=128, 6 blocks | 1.76 ms | **3.13 ms** | 1.78× | 0% | 1.88 ms | **3.19 ms** |
| conv-2D simple | Float64 190×200 ⋆ 20×10 → 210×210 | 1.54 ms | **3.96 ms** | 2.58× | 1% | 1.77 ms | **4.16 ms** |
| conv-2D overlapsave | Float64 190×200 ⋆ 20×10 → nfft=128, 4 blocks | 647.1 µs | **1.27 ms** | 1.95× | 1% | 693.5 µs | **1.33 ms** |
| conv-2D simple | ComplexF64 190×200 ⋆ 20×10 → 210×210 | 3.01 ms | **8.37 ms** | 2.78× | 1% | 3.94 ms | **8.57 ms** |
| conv-2D overlapsave | ComplexF64 190×200 ⋆ 20×10 → nfft=128, 4 blocks | 1.22 ms | **2.17 ms** | 1.78× | 0% | 1.27 ms | **2.25 ms** |
| conv-2D simple | Float64 190×200 ⋆ 210×200 → 400×400 | 4.47 ms | **9.04 ms** | 2.02× | 0% | 5.37 ms | **9.32 ms** |
| conv-2D overlapsave | Float64 190×200 ⋆ 210×200 → nfft=420, 4 blocks | 18.59 ms | **55.33 ms** | 2.98× | 0% | 20.82 ms | **55.91 ms** |
| conv-2D simple | ComplexF64 190×200 ⋆ 210×200 → 400×400 | 8.12 ms | **17.42 ms** | 2.15× | 0% | 8.97 ms | **19.89 ms** |
| conv-2D overlapsave | ComplexF64 190×200 ⋆ 210×200 → nfft=420, 4 blocks | 35.72 ms | **99.00 ms** | 2.77× | 0% | 37.60 ms | **99.13 ms** |
| conv-ND small | Float64 5 ⋆ 5 → 9 | 4.5 µs | **2.2 µs** | 0.48× | 74% | 25.9 µs | **28.9 µs** |
| conv-ND small | Float64 5×5 ⋆ 5×5 → 9×9 | 15.6 µs | **16.8 µs** | 1.07× | 65% | 58.0 µs | **76.5 µs** |
| conv-ND small | Float64 5×5×5 ⋆ 5×5×5 → 9×9×9 | 51.5 µs | **80.9 µs** | 1.57× | 15% | 162.2 µs | **1.65 s** |
| conv-ND small | Float64 3×3×6 ⋆ 2×2×2 → 4×4×7 | 38.8 µs | **23.0 µs** | 0.59× | 45% | 69.6 µs | **37.1 µs** |
| conv-ND small | Float64 2×2×2×2×2×2 ⋆ 1×1×1×1×1×1 → 2×2×2×2×2×2 | 107.9 µs | **48.8 µs** | 0.45× | 29% | 610.8 µs | **3.12 s** |
| conv-ND small | Float64 4×7×1 ⋆ 3×3×3 → 6×9×3 | 40.0 µs | **27.8 µs** | 0.70× | 40% | 79.4 µs | **46.6 µs** |
| os-test overlapsave | Float32 128^1 ⋆ 12^1, nfft=64, 3 blocks | 11.6 µs | **3.3 µs** | 0.29× | 43% | 270.7 µs | **24.0 µs** |
| os-test reference | Float32 128^1 ⋆ 12^1 → 140^1 | 40.3 µs | **13.8 µs** | 0.34× | 58% | 522.7 µs | **48.37 ms** |
| os-test overlapsave | Float32 128^1 ⋆ 128^1, nfft=256, 2 blocks | 43.1 µs | **8.4 µs** | 0.19× | 50% | 790.2 µs | **74.5 µs** |
| os-test reference | Float32 128^1 ⋆ 128^1 → 256^1 | 42.6 µs | **7.8 µs** | 0.18× | 59% | 53.2 µs | **110.4 µs** |
| os-test overlapsave | Float64 128^1 ⋆ 12^1, nfft=64, 3 blocks | 11.8 µs | **3.9 µs** | 0.33× | 45% | 36.5 µs | **18.4 µs** |
| os-test reference | Float64 128^1 ⋆ 12^1 → 140^1 | 41.6 µs | **17.0 µs** | 0.41× | 59% | 416.0 µs | **217.8 µs** |
| os-test overlapsave | Float64 128^1 ⋆ 128^1, nfft=256, 2 blocks | 30.8 µs | **12.8 µs** | 0.41× | 50% | 761.8 µs | **27.3 µs** |
| os-test reference | Float64 128^1 ⋆ 128^1 → 256^1 | 30.0 µs | **12.6 µs** | 0.42× | 55% | 42.2 µs | **27.4 µs** |
| os-test overlapsave | ComplexF64 128^1 ⋆ 12^1, nfft=64, 3 blocks | 22.4 µs | **6.0 µs** | 0.27× | 45% | 43.5 µs | **32.4 µs** |
| os-test reference | ComplexF64 128^1 ⋆ 12^1 → 140^1 | 23.2 µs | **22.4 µs** | 0.97× | 62% | 262.0 µs | **70.3 µs** |
| os-test overlapsave | ComplexF64 128^1 ⋆ 128^1, nfft=256, 2 blocks | 28.3 µs | **15.4 µs** | 0.54× | 34% | 404.5 µs | **33.0 µs** |
| os-test reference | ComplexF64 128^1 ⋆ 128^1 → 256^1 | 25.1 µs | **11.6 µs** | 0.46× | 47% | 41.2 µs | **29.0 µs** |
| os-test overlapsave | Float32 128^2 ⋆ 12^2, nfft=64, 9 blocks | 263.2 µs | **636.7 µs** | 2.42× | 2% | 374.9 µs | **43.31 s** |
| os-test reference | Float32 128^2 ⋆ 12^2 → 140^2 | 408.6 µs | **1.36 ms** | 3.32× | 2% | 707.5 µs | **119.28 ms** |
| os-test overlapsave | Float32 128^2 ⋆ 128^2, nfft=256, 4 blocks | 3.72 ms | **3.82 ms** | 1.03× | 0% | 4.14 ms | **3.82 ms** |
| os-test reference | Float32 128^2 ⋆ 128^2 → 256^2 | 1.32 ms | **1.29 ms** | 0.98× | 1% | 1.35 ms | **1.39 ms** |
| os-test overlapsave | Float64 128^2 ⋆ 12^2, nfft=64, 9 blocks | 300.5 µs | **696.8 µs** | 2.32× | 2% | 377.4 µs | **730.9 µs** |
| os-test reference | Float64 128^2 ⋆ 12^2 → 140^2 | 514.5 µs | **1.56 ms** | 3.03× | 2% | 785.4 µs | **1.59 ms** |
| os-test overlapsave | Float64 128^2 ⋆ 128^2, nfft=256, 4 blocks | 4.60 ms | **5.44 ms** | 1.18× | 0% | 5.05 ms | **5.40 ms** |
| os-test reference | Float64 128^2 ⋆ 128^2 → 256^2 | 1.62 ms | **1.83 ms** | 1.13× | 1% | 1.78 ms | **2.19 ms** |
| os-test overlapsave | ComplexF64 128^2 ⋆ 12^2, nfft=64, 9 blocks | 510.8 µs | **1.18 ms** | 2.31× | 0% | 598.8 µs | **1.22 ms** |
| os-test reference | ComplexF64 128^2 ⋆ 12^2 → 140^2 | 971.6 µs | **2.50 ms** | 2.58× | 1% | 1.68 ms | **2.58 ms** |
| os-test overlapsave | ComplexF64 128^2 ⋆ 128^2, nfft=256, 4 blocks | 11.76 ms | **12.10 ms** | 1.03× | 0% | 12.57 ms | **12.30 ms** |
| os-test reference | ComplexF64 128^2 ⋆ 128^2 → 256^2 | 3.99 ms | **4.18 ms** | 1.05× | 0% | 4.09 ms | **4.26 ms** |
| os-test overlapsave | Float32 128^3 ⋆ 12^3, nfft=64, 27 blocks | 81.08 ms | **199.08 ms** | 2.46× | 0% | 81.53 ms | **58.04 s** |
| os-test reference | Float32 128^3 ⋆ 12^3 → 140^3 | 86.79 ms | **303.04 ms** | 3.49× | 0% | 92.72 ms | **488.52 ms** |
| os-test overlapsave | Float64 128^3 ⋆ 12^3, nfft=64, 27 blocks | 109.40 ms | **232.14 ms** | 2.12× | 0% | 110.91 ms | **237.23 ms** |
| os-test reference | Float64 128^3 ⋆ 12^3 → 140^3 | 150.72 ms | **380.20 ms** | 2.52× | 0% | 304.16 ms | **389.03 ms** |
| os-test overlapsave | ComplexF64 128^3 ⋆ 12^3, nfft=64, 27 blocks | 194.19 ms | **430.43 ms** | 2.22× | 0% | 196.35 ms | **622.25 ms** |
| os-test reference | ComplexF64 128^3 ⋆ 12^3 → 140^3 | 226.17 ms | **675.25 ms** | 2.99× | 0% | 228.15 ms | **701.10 ms** |
| os-test adversarial | Float64 128 ⋆ 12, nfft=256, 1 blocks | 28.3 µs | **11.0 µs** | 0.39× | 54% | 53.5 µs | **297.4 µs** |
| os-test adversarial | Float64 128 ⋆ 13, nfft=32, 7 blocks | 11.9 µs | **3.7 µs** | 0.31× | 32% | 135.8 µs | **12.4 µs** |
| os-test adversarial | Float64 128 ⋆ 12, nfft=32, 7 blocks | 11.9 µs | **3.8 µs** | 0.32× | 32% | 22.2 µs | **12.6 µs** |
| os-test adversarial | Float64 25 ⋆ 4, nfft=16, 3 blocks | 4.7 µs | **1.9 µs** | 0.40× | 55% | 32.1 µs | **10.0 µs** |

## Totals per group

| group | FFTW steady | FFTA steady | FFTA / FFTW | FFTW first calls | FFTA first calls | first-call ratio |
|:--|--:|--:|--:|--:|--:|--:|
| conv-1D simple | 252.0 µs | **185.5 µs** | 0.74× | 2.62 ms | **169.92 ms** | 64.93× |
| conv-1D overlapsave | 167.0 µs | **95.8 µs** | 0.57× | 628.2 µs | **227.0 µs** | 0.36× |
| conv-2D simple | 21.00 ms | **46.23 ms** | 2.20× | 28.82 ms | **1.08 s** | 37.53× |
| conv-2D overlapsave | 59.13 ms | **163.09 ms** | 2.76× | 65.89 ms | **164.21 ms** | 2.49× |
| conv-ND small | 258.3 µs | **199.4 µs** | 0.77× | 1.01 ms | **4.77 s** | 4746.28× |
| os-test overlapsave | 405.97 ms | **885.58 ms** | 2.18× | 414.22 ms | **102.23 s** | 246.81× |
| os-test reference | 472.70 ms | **1.37 s** | 2.90× | 636.76 ms | **1.76 s** | 2.76× |
| os-test adversarial | 56.8 µs | **20.4 µs** | 0.36× | 243.5 µs | **332.5 µs** | 1.37× |
| **all** | 959.53 ms | **2.47 s** | 2.57× | 1.15 s | **110.18 s** | 95.80× |

## N-d transforms, one dimension at a time (`nd_stages.jl`)

`FFTA 1-D × pencils` = one 1-D FFTA transform of the pencil length times the number of pencils; `pencil overhead` = the pass divided by that.

| shape | pass | FFTW | FFTA | FFTA / FFTW | FFTA 1-D × pencils | pencil overhead |
|:--|:--|--:|--:|--:|--:|--:|
| ComplexF64 140×140 | dim 1 (140 pencils of 140) | 110.7 µs | **391.8 µs** | 3.54× | 324.8 µs | 1.21× |
| ComplexF64 140×140 | dim 2 (140 pencils of 140) | 121.3 µs | **421.6 µs** | 3.48× | 330.4 µs | 1.28× |
| ComplexF64 140×140 | **all dims** | 246.4 µs | **765.3 µs** | 3.11× | | |
| Float64 140×140 | dim 1 (140 pencils of 140) | 56.8 µs | **253.3 µs** | 4.46× | 246.4 µs | 1.03× |
| Float64 140×140 | **all dims** | 122.1 µs | **447.2 µs** | 3.66× | | |
| ComplexF64 256×256 | dim 1 (256 pencils of 256) | 371.6 µs | **409.6 µs** | 1.10× | 368.6 µs | 1.11× |
| ComplexF64 256×256 | dim 2 (256 pencils of 256) | 823.7 µs | **1.04 ms** | 1.26× | 368.6 µs | 2.82× |
| ComplexF64 256×256 | **all dims** | 1.17 ms | **1.22 ms** | 1.05× | | |
| Float64 256×256 | dim 1 (256 pencils of 256) | 185.9 µs | **239.9 µs** | 1.29× | 225.3 µs | 1.06× |
| Float64 256×256 | **all dims** | 413.5 µs | **465.9 µs** | 1.13× | | |
| ComplexF64 140×140×140 | dim 1 (19600 pencils of 140) | 21.28 ms | **66.45 ms** | 3.12× | 47.04 ms | 1.41× |
| ComplexF64 140×140×140 | dim 2 (19600 pencils of 140) | 27.15 ms | **84.44 ms** | 3.11× | 46.26 ms | 1.83× |
| ComplexF64 140×140×140 | dim 3 (19600 pencils of 140) | 28.08 ms | **87.55 ms** | 3.12× | 44.69 ms | 1.96× |
| ComplexF64 140×140×140 | **all dims** | 66.95 ms | **207.27 ms** | 3.10× | | |
| Float64 140×140×140 | dim 1 (19600 pencils of 140) | 8.38 ms | **38.48 ms** | 4.59× | 35.28 ms | 1.09× |
| Float64 140×140×140 | **all dims** | 30.97 ms | **110.50 ms** | 3.57× | | |
| ComplexF64 64×64×64 | dim 1 (4096 pencils of 64) | 655.8 µs | **1.65 ms** | 2.51× | 1.47 ms | 1.12× |
| ComplexF64 64×64×64 | dim 2 (4096 pencils of 64) | 1.08 ms | **2.34 ms** | 2.17× | 1.47 ms | 1.59× |
| ComplexF64 64×64×64 | dim 3 (4096 pencils of 64) | 1.32 ms | **2.51 ms** | 1.89× | 1.47 ms | 1.70× |
| ComplexF64 64×64×64 | **all dims** | 2.50 ms | **6.62 ms** | 2.65× | | |
| Float64 64×64×64 | dim 1 (4096 pencils of 64) | 447.4 µs | **1.09 ms** | 2.44× | 983.0 µs | 1.11× |
| Float64 64×64×64 | **all dims** | 1.37 ms | **3.36 ms** | 2.45× | | |
