# DSP.jl's convolution workload on FFTA vs FFTW

`dsp_cases.jl` replays the FFT call sequences of DSP.jl's `conv` kernels (one-shot, planning every time, as DSP does) over the size and element-type patterns of `test/dsp.jl`, in one process per implementation; `nd_stages.jl` times an N-d transform one dimension at a time. aarch64 Neoverse-N1, 1 thread, FFTA `main` (with the `Vector`-only kernel boundary), full run.

**Reading:** steady-state execution of the whole replay is 1.05× of FFTW (2-D/3-D convolutions at 7-smooth sizes are 2–2.5×; the small one-shot cases are *faster* than FFTW — its one-shot planning costs more). First calls total 32 s vs 27 s (down from 110 s): the kernels now see only `Vector` arguments, so a new (element type, dimensionality) no longer recompiles them; the residual is one 6-D case (~3 s of plan-level code) and the one-shot `irfft`/composite paths (~0.2 s each). The DSP.jl test suite itself runs 488 s on FFTA vs 431 s on FFTW (was 787 s).

| group | case | FFTW steady | FFTA steady | FFTA / FFTW | FFTA plans | FFTW first call | FFTA first call |
|:--|:--|--:|--:|--:|--:|--:|--:|
| conv-1D simple | Float64 10⋆10 → n=20 | 10.5 µs | **3.2 µs** | 0.31× | 71% | 60.8 µs | **48.91 ms** |
| conv-1D overlapsave | Float64 10⋆10 → nfft=20, 2 blocks | 10.7 µs | **3.2 µs** | 0.29× | 62% | 42.5 µs | **26.4 µs** |
| conv-1D simple | ComplexF64 10⋆10 → n=20 | 4.7 µs | **4.2 µs** | 0.89× | 77% | 565.7 µs | **121.92 ms** |
| conv-1D overlapsave | ComplexF64 10⋆10 → nfft=20, 2 blocks | 4.8 µs | **4.8 µs** | 0.98× | 68% | 22.8 µs | **40.5 µs** |
| conv-1D simple | Float64 10⋆200 → n=210 | 37.4 µs | **20.2 µs** | 0.54× | 61% | 247.8 µs | **66.1 µs** |
| conv-1D overlapsave | Float64 10⋆200 → nfft=64, 4 blocks | 12.0 µs | **4.4 µs** | 0.37× | 41% | 174.4 µs | **13.2 µs** |
| conv-1D simple | ComplexF64 10⋆200 → n=210 | 43.9 µs | **42.5 µs** | 0.97× | 55% | 272.1 µs | **75.0 µs** |
| conv-1D overlapsave | ComplexF64 10⋆200 → nfft=64, 4 blocks | 23.2 µs | **6.8 µs** | 0.29× | 38% | 199.7 µs | **18.3 µs** |
| conv-1D simple | Float64 200⋆10 → n=210 | 37.6 µs | **20.6 µs** | 0.55× | 60% | 53.8 µs | **40.1 µs** |
| conv-1D overlapsave | Float64 200⋆10 → nfft=64, 4 blocks | 12.0 µs | **4.4 µs** | 0.37× | 41% | 26.2 µs | **15.3 µs** |
| conv-1D simple | ComplexF64 200⋆10 → n=210 | 43.9 µs | **42.8 µs** | 0.97× | 55% | 84.8 µs | **73.8 µs** |
| conv-1D overlapsave | ComplexF64 200⋆10 → nfft=64, 4 blocks | 23.5 µs | **7.0 µs** | 0.30× | 37% | 38.8 µs | **19.2 µs** |
| conv-1D simple | Float64 200⋆200 → n=400 | 44.6 µs | **22.1 µs** | 0.50× | 50% | 1.02 ms | **48.7 µs** |
| conv-1D overlapsave | Float64 200⋆200 → nfft=400, 2 blocks | 46.7 µs | **26.2 µs** | 0.56× | 41% | 64.0 µs | **38.0 µs** |
| conv-1D simple | ComplexF64 200⋆200 → n=400 | 29.2 µs | **27.5 µs** | 0.94× | 33% | 322.4 µs | **49.1 µs** |
| conv-1D overlapsave | ComplexF64 200⋆200 → nfft=400, 2 blocks | 33.8 µs | **35.9 µs** | 1.06× | 26% | 47.0 µs | **45.1 µs** |
| conv-2D simple | Float64 10×20 ⋆ 20×10 → 30×30 | 71.3 µs | **79.6 µs** | 1.12× | 20% | 429.0 µs | **61.51 ms** |
| conv-2D overlapsave | Float64 10×20 ⋆ 20×10 → nfft=40, 2 blocks | 102.8 µs | **132.7 µs** | 1.29× | 10% | 1.35 ms | **213.6 µs** |
| conv-2D simple | ComplexF64 10×20 ⋆ 20×10 → 30×30 | 91.0 µs | **141.8 µs** | 1.56× | 7% | 922.6 µs | **118.54 ms** |
| conv-2D overlapsave | ComplexF64 10×20 ⋆ 20×10 → nfft=40, 2 blocks | 163.2 µs | **200.6 µs** | 1.23× | 2% | 1.20 ms | **254.2 µs** |
| conv-2D simple | Float64 10×20 ⋆ 210×200 → 224×224 | 1.20 ms | **2.12 ms** | 1.76× | 1% | 2.15 ms | **2.22 ms** |
| conv-2D overlapsave | Float64 10×20 ⋆ 210×200 → nfft=128, 6 blocks | 933.4 µs | **1.78 ms** | 1.91× | 1% | 1.02 ms | **1.85 ms** |
| conv-2D simple | ComplexF64 10×20 ⋆ 210×200 → 224×224 | 2.47 ms | **4.26 ms** | 1.73× | 0% | 4.06 ms | **4.28 ms** |
| conv-2D overlapsave | ComplexF64 10×20 ⋆ 210×200 → nfft=128, 6 blocks | 1.79 ms | **3.35 ms** | 1.87× | 0% | 1.91 ms | **3.46 ms** |
| conv-2D simple | Float64 190×200 ⋆ 20×10 → 210×210 | 1.53 ms | **3.55 ms** | 2.33× | 1% | 1.78 ms | **3.60 ms** |
| conv-2D overlapsave | Float64 190×200 ⋆ 20×10 → nfft=128, 4 blocks | 648.6 µs | **1.23 ms** | 1.90× | 1% | 678.2 µs | **1.29 ms** |
| conv-2D simple | ComplexF64 190×200 ⋆ 20×10 → 210×210 | 3.03 ms | **7.78 ms** | 2.57× | 1% | 3.90 ms | **7.83 ms** |
| conv-2D overlapsave | ComplexF64 190×200 ⋆ 20×10 → nfft=128, 4 blocks | 1.24 ms | **2.33 ms** | 1.88× | 0% | 1.29 ms | **2.38 ms** |
| conv-2D simple | Float64 190×200 ⋆ 210×200 → 400×400 | 4.33 ms | **7.74 ms** | 1.79× | 0% | 4.81 ms | **8.01 ms** |
| conv-2D overlapsave | Float64 190×200 ⋆ 210×200 → nfft=420, 4 blocks | 18.60 ms | **47.20 ms** | 2.54× | 0% | 20.63 ms | **49.27 ms** |
| conv-2D simple | ComplexF64 190×200 ⋆ 210×200 → 400×400 | 8.13 ms | **15.51 ms** | 1.91× | 0% | 9.23 ms | **15.92 ms** |
| conv-2D overlapsave | ComplexF64 190×200 ⋆ 210×200 → nfft=420, 4 blocks | 35.63 ms | **88.93 ms** | 2.50× | 0% | 37.25 ms | **89.79 ms** |
| conv-ND small | Float64 5 ⋆ 5 → 9 | 4.5 µs | **2.2 µs** | 0.48× | 74% | 22.2 µs | **25.2 µs** |
| conv-ND small | Float64 5×5 ⋆ 5×5 → 9×9 | 15.7 µs | **16.2 µs** | 1.03× | 67% | 50.9 µs | **59.4 µs** |
| conv-ND small | Float64 5×5×5 ⋆ 5×5×5 → 9×9×9 | 50.4 µs | **70.6 µs** | 1.40× | 17% | 156.2 µs | **179.41 ms** |
| conv-ND small | Float64 3×3×6 ⋆ 2×2×2 → 4×4×7 | 38.8 µs | **20.8 µs** | 0.54× | 49% | 67.9 µs | **32.8 µs** |
| conv-ND small | Float64 2×2×2×2×2×2 ⋆ 1×1×1×1×1×1 → 2×2×2×2×2×2 | 108.4 µs | **47.4 µs** | 0.44× | 30% | 645.4 µs | **2.94 s** |
| conv-ND small | Float64 4×7×1 ⋆ 3×3×3 → 6×9×3 | 39.8 µs | **25.3 µs** | 0.64× | 43% | 68.0 µs | **43.6 µs** |
| os-test overlapsave | Float32 128^1 ⋆ 12^1, nfft=64, 3 blocks | 11.7 µs | **3.3 µs** | 0.28× | 45% | 303.5 µs | **24.1 µs** |
| os-test reference | Float32 128^1 ⋆ 12^1 → 140^1 | 40.5 µs | **12.9 µs** | 0.32× | 63% | 559.3 µs | **45.74 ms** |
| os-test overlapsave | Float32 128^1 ⋆ 128^1, nfft=256, 2 blocks | 43.2 µs | **7.6 µs** | 0.18× | 53% | 817.5 µs | **69.7 µs** |
| os-test reference | Float32 128^1 ⋆ 128^1 → 256^1 | 42.2 µs | **7.2 µs** | 0.17× | 61% | 53.0 µs | **22.1 µs** |
| os-test overlapsave | Float64 128^1 ⋆ 12^1, nfft=64, 3 blocks | 11.8 µs | **3.9 µs** | 0.33× | 47% | 39.0 µs | **15.5 µs** |
| os-test reference | Float64 128^1 ⋆ 12^1 → 140^1 | 41.4 µs | **16.2 µs** | 0.39× | 60% | 616.6 µs | **45.0 µs** |
| os-test overlapsave | Float64 128^1 ⋆ 128^1, nfft=256, 2 blocks | 30.9 µs | **12.4 µs** | 0.40× | 50% | 753.7 µs | **23.9 µs** |
| os-test reference | Float64 128^1 ⋆ 128^1 → 256^1 | 29.8 µs | **11.2 µs** | 0.38× | 58% | 40.9 µs | **24.0 µs** |
| os-test overlapsave | ComplexF64 128^1 ⋆ 12^1, nfft=64, 3 blocks | 22.4 µs | **5.8 µs** | 0.26× | 46% | 44.9 µs | **31.7 µs** |
| os-test reference | ComplexF64 128^1 ⋆ 12^1 → 140^1 | 23.1 µs | **22.3 µs** | 0.97× | 62% | 255.0 µs | **40.2 µs** |
| os-test overlapsave | ComplexF64 128^1 ⋆ 128^1, nfft=256, 2 blocks | 28.2 µs | **14.4 µs** | 0.51× | 37% | 263.7 µs | **34.8 µs** |
| os-test reference | ComplexF64 128^1 ⋆ 128^1 → 256^1 | 25.0 µs | **10.9 µs** | 0.43× | 49% | 41.2 µs | **24.8 µs** |
| os-test overlapsave | Float32 128^2 ⋆ 12^2, nfft=64, 9 blocks | 262.6 µs | **606.7 µs** | 2.31× | 2% | 369.0 µs | **768.8 µs** |
| os-test reference | Float32 128^2 ⋆ 12^2 → 140^2 | 410.0 µs | **1.16 ms** | 2.83× | 2% | 691.5 µs | **53.76 ms** |
| os-test overlapsave | Float32 128^2 ⋆ 128^2, nfft=256, 4 blocks | 3.72 ms | **3.71 ms** | 1.00× | 0% | 4.14 ms | **3.78 ms** |
| os-test reference | Float32 128^2 ⋆ 128^2 → 256^2 | 1.31 ms | **1.25 ms** | 0.96× | 1% | 1.33 ms | **1.34 ms** |
| os-test overlapsave | Float64 128^2 ⋆ 12^2, nfft=64, 9 blocks | 299.8 µs | **667.8 µs** | 2.23× | 2% | 373.0 µs | **693.2 µs** |
| os-test reference | Float64 128^2 ⋆ 12^2 → 140^2 | 513.1 µs | **1.33 ms** | 2.59× | 2% | 796.7 µs | **3.74 ms** |
| os-test overlapsave | Float64 128^2 ⋆ 128^2, nfft=256, 4 blocks | 4.66 ms | **5.30 ms** | 1.14× | 0% | 5.05 ms | **5.34 ms** |
| os-test reference | Float64 128^2 ⋆ 128^2 → 256^2 | 1.57 ms | **1.77 ms** | 1.13× | 1% | 1.60 ms | **1.87 ms** |
| os-test overlapsave | ComplexF64 128^2 ⋆ 12^2, nfft=64, 9 blocks | 510.4 µs | **1.21 ms** | 2.36× | 0% | 617.4 µs | **1.25 ms** |
| os-test reference | ComplexF64 128^2 ⋆ 12^2 → 140^2 | 966.0 µs | **2.28 ms** | 2.36× | 1% | 1.69 ms | **2.37 ms** |
| os-test overlapsave | ComplexF64 128^2 ⋆ 128^2, nfft=256, 4 blocks | 11.80 ms | **11.97 ms** | 1.01× | 0% | 12.62 ms | **12.35 ms** |
| os-test reference | ComplexF64 128^2 ⋆ 128^2 → 256^2 | 3.99 ms | **4.01 ms** | 1.00× | 0% | 4.10 ms | **4.45 ms** |
| os-test overlapsave | Float32 128^3 ⋆ 12^3, nfft=64, 27 blocks | 80.72 ms | **181.87 ms** | 2.25× | 0% | 81.44 ms | **184.01 ms** |
| os-test reference | Float32 128^3 ⋆ 12^3 → 140^3 | 89.96 ms | **253.75 ms** | 2.82× | 0% | 88.11 ms | **338.04 ms** |
| os-test overlapsave | Float32 128^3 ⋆ 128^3, nfft=256, 8 blocks | 3.88 s | **3.78 s** | 0.97× | 0% | 3.88 s | **3.97 s** |
| os-test reference | Float32 128^3 ⋆ 128^3 → 256^3 | 808.05 ms | **704.09 ms** | 0.87× | 0% | 1.08 s | **939.87 ms** |
| os-test overlapsave | Float64 128^3 ⋆ 12^3, nfft=64, 27 blocks | 109.31 ms | **221.82 ms** | 2.03× | 0% | 110.51 ms | **219.85 ms** |
| os-test reference | Float64 128^3 ⋆ 12^3 → 140^3 | 132.44 ms | **310.49 ms** | 2.34× | 0% | 179.63 ms | **322.20 ms** |
| os-test overlapsave | Float64 128^3 ⋆ 128^3, nfft=256, 8 blocks | 5.33 s | **5.71 s** | 1.07× | 0% | 5.48 s | **6.12 s** |
| os-test reference | Float64 128^3 ⋆ 128^3 → 256^3 | 1.34 s | **1.11 s** | 0.83× | 0% | 1.48 s | **1.34 s** |
| os-test overlapsave | ComplexF64 128^3 ⋆ 12^3, nfft=64, 27 blocks | 187.36 ms | **420.64 ms** | 2.25× | 0% | 187.95 ms | **516.79 ms** |
| os-test reference | ComplexF64 128^3 ⋆ 12^3 → 140^3 | 223.98 ms | **545.83 ms** | 2.44× | 0% | 225.18 ms | **568.61 ms** |
| os-test overlapsave | ComplexF64 128^3 ⋆ 128^3, nfft=256, 8 blocks | 11.51 s | **11.48 s** | 1.00× | 0% | 11.56 s | **11.51 s** |
| os-test reference | ComplexF64 128^3 ⋆ 128^3 → 256^3 | 2.05 s | **2.09 s** | 1.02× | 0% | 2.07 s | **2.13 s** |
| os-test adversarial | Float64 128 ⋆ 12, nfft=256, 1 blocks | 28.4 µs | **9.6 µs** | 0.34× | 76% | 59.2 µs | **23.3 µs** |
| os-test adversarial | Float64 128 ⋆ 13, nfft=32, 7 blocks | 12.0 µs | **3.5 µs** | 0.29× | 35% | 135.5 µs | **11.9 µs** |
| os-test adversarial | Float64 128 ⋆ 12, nfft=32, 7 blocks | 12.0 µs | **3.6 µs** | 0.30× | 35% | 20.0 µs | **12.2 µs** |
| os-test adversarial | Float64 25 ⋆ 4, nfft=16, 3 blocks | 4.7 µs | **1.8 µs** | 0.37× | 59% | 29.4 µs | **9.2 µs** |

## Totals per group

| group | FFTW steady | FFTA steady | FFTA / FFTW | FFTW first calls | FFTA first calls | first-call ratio |
|:--|--:|--:|--:|--:|--:|--:|
| conv-1D simple | 251.8 µs | **183.0 µs** | 0.73× | 2.63 ms | **171.19 ms** | 65.06× |
| conv-1D overlapsave | 166.8 µs | **92.6 µs** | 0.56× | 615.4 µs | **216.1 µs** | 0.35× |
| conv-2D simple | 20.85 ms | **41.18 ms** | 1.97× | 27.27 ms | **221.92 ms** | 8.14× |
| conv-2D overlapsave | 59.12 ms | **145.16 ms** | 2.46× | 65.32 ms | **148.51 ms** | 2.27× |
| conv-ND small | 257.6 µs | **182.5 µs** | 0.71× | 1.01 ms | **3.11 s** | 3081.82× |
| os-test overlapsave | 21.12 s | **21.82 s** | 1.03× | 21.32 s | **22.55 s** | 1.06× |
| os-test reference | 4.65 s | **5.02 s** | 1.08× | 5.14 s | **5.75 s** | 1.12× |
| os-test adversarial | 57.1 µs | **18.4 µs** | 0.32× | 244.1 µs | **56.6 µs** | 0.23× |
| **all** | 25.85 s | **27.02 s** | 1.05× | 26.56 s | **31.95 s** | 1.20× |

## N-d transforms, one dimension at a time (`nd_stages.jl`)

`FFTA 1-D × pencils` = one 1-D FFTA transform of the pencil length times the number of pencils; `pencil overhead` = the pass divided by that.

| shape | pass | FFTW | FFTA | FFTA / FFTW | FFTA 1-D × pencils | pencil overhead |
|:--|:--|--:|--:|--:|--:|--:|
| ComplexF64 140×140 | dim 1 (140 pencils of 140) | 112.0 µs | **346.5 µs** | 3.09× | 324.8 µs | 1.07× |
| ComplexF64 140×140 | dim 2 (140 pencils of 140) | 125.8 µs | **358.3 µs** | 2.85× | 319.2 µs | 1.12× |
| ComplexF64 140×140 | **all dims** | 246.6 µs | **721.1 µs** | 2.92× | | |
| Float64 140×140 | dim 1 (140 pencils of 140) | 57.0 µs | **217.5 µs** | 3.82× | 212.8 µs | 1.02× |
| Float64 140×140 | **all dims** | 122.2 µs | **394.4 µs** | 3.23× | | |
| ComplexF64 256×256 | dim 1 (256 pencils of 256) | 369.6 µs | **445.2 µs** | 1.20× | 368.6 µs | 1.21× |
| ComplexF64 256×256 | dim 2 (256 pencils of 256) | 830.0 µs | **732.7 µs** | 0.88× | 368.6 µs | 1.99× |
| ComplexF64 256×256 | **all dims** | 1.20 ms | **1.19 ms** | 0.99× | | |
| Float64 256×256 | dim 1 (256 pencils of 256) | 184.8 µs | **251.7 µs** | 1.36× | 235.5 µs | 1.07× |
| Float64 256×256 | **all dims** | 411.9 µs | **498.7 µs** | 1.21× | | |
| ComplexF64 140×140×140 | dim 1 (19600 pencils of 140) | 22.41 ms | **54.81 ms** | 2.45× | 45.47 ms | 1.21× |
| ComplexF64 140×140×140 | dim 2 (19600 pencils of 140) | 30.97 ms | **60.57 ms** | 1.96× | 45.47 ms | 1.33× |
| ComplexF64 140×140×140 | dim 3 (19600 pencils of 140) | 29.62 ms | **63.57 ms** | 2.15× | 45.47 ms | 1.40× |
| ComplexF64 140×140×140 | **all dims** | 90.16 ms | **181.18 ms** | 2.01× | | |
| Float64 140×140×140 | dim 1 (19600 pencils of 140) | 8.72 ms | **33.78 ms** | 3.88× | 29.79 ms | 1.13× |
| Float64 140×140×140 | **all dims** | 34.86 ms | **103.59 ms** | 2.97× | | |
| ComplexF64 64×64×64 | dim 1 (4096 pencils of 64) | 655.1 µs | **1.77 ms** | 2.71× | 1.47 ms | 1.20× |
| ComplexF64 64×64×64 | dim 2 (4096 pencils of 64) | 1.09 ms | **2.22 ms** | 2.03× | 1.47 ms | 1.50× |
| ComplexF64 64×64×64 | dim 3 (4096 pencils of 64) | 1.37 ms | **2.69 ms** | 1.97× | 1.47 ms | 1.83× |
| ComplexF64 64×64×64 | **all dims** | 2.51 ms | **6.61 ms** | 2.63× | | |
| Float64 64×64×64 | dim 1 (4096 pencils of 64) | 445.3 µs | **1.13 ms** | 2.54× | 983.0 µs | 1.15× |
| Float64 64×64×64 | **all dims** | 1.37 ms | **3.18 ms** | 2.33× | | |
