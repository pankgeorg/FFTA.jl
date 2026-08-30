# DSP.jl's convolution workload on FFTA vs FFTW

`dsp_cases.jl` replays the FFT call sequences of DSP.jl's `conv` kernels (one-shot, planning
every time, as DSP does) over the size and element-type patterns of `test/dsp.jl`, in one
process per implementation; `nd_stages.jl` times an N-d transform one dimension at a time.
aarch64 Neoverse-N1, 1 thread, FFTA `main` (Stockham engine + session caches + adjoint stack),
full run. `steady` = minimum of repeated runs; `first call` = the first run of that case in a
fresh process, which includes compiling whatever its element type, shape and size need (a test
suite pays this once per combination). `plans` = share of FFTA's steady time spent constructing
plans.

| group | case | FFTW steady | FFTA steady | FFTA / FFTW | FFTA plans | FFTW first call | FFTA first call |
|:--|:--|--:|--:|--:|--:|--:|--:|
| conv-1D simple | Float64 10⋆10 → n=20 | 10.6 µs | **3.2 µs** | 0.30× | 70% | 64.7 µs | **48.36 ms** |
| conv-1D overlapsave | Float64 10⋆10 → nfft=20, 2 blocks | 10.8 µs | **3.3 µs** | 0.30× | 60% | 41.5 µs | **24.3 µs** |
| conv-1D simple | ComplexF64 10⋆10 → n=20 | 4.8 µs | **4.3 µs** | 0.90× | 77% | 562.3 µs | **111.01 ms** |
| conv-1D overlapsave | ComplexF64 10⋆10 → nfft=20, 2 blocks | 4.9 µs | **4.9 µs** | 1.01× | 67% | 22.4 µs | **42.4 µs** |
| conv-1D simple | Float64 10⋆200 → n=210 | 37.4 µs | **8.6 µs** | 0.23× | 42% | 240.0 µs | **63.2 µs** |
| conv-1D overlapsave | Float64 10⋆200 → nfft=64, 4 blocks | 12.0 µs | **4.5 µs** | 0.38× | 42% | 175.0 µs | **15.3 µs** |
| conv-1D simple | ComplexF64 10⋆200 → n=210 | 43.6 µs | **9.9 µs** | 0.23× | 27% | 282.1 µs | **52.5 µs** |
| conv-1D overlapsave | ComplexF64 10⋆200 → nfft=64, 4 blocks | 23.4 µs | **7.3 µs** | 0.31× | 40% | 289.3 µs | **22.0 µs** |
| conv-1D simple | Float64 200⋆10 → n=210 | 37.6 µs | **8.6 µs** | 0.23× | 42% | 53.4 µs | **27.4 µs** |
| conv-1D overlapsave | Float64 200⋆10 → nfft=64, 4 blocks | 12.1 µs | **4.5 µs** | 0.37× | 42% | 24.3 µs | **15.8 µs** |
| conv-1D simple | ComplexF64 200⋆10 → n=210 | 43.7 µs | **9.9 µs** | 0.23× | 26% | 85.2 µs | **36.7 µs** |
| conv-1D overlapsave | ComplexF64 200⋆10 → nfft=64, 4 blocks | 23.6 µs | **7.3 µs** | 0.31× | 37% | 41.2 µs | **20.4 µs** |
| conv-1D simple | Float64 200⋆200 → n=400 | 44.7 µs | **14.8 µs** | 0.33× | 42% | 1.03 ms | **38.5 µs** |
| conv-1D overlapsave | Float64 200⋆200 → nfft=400, 2 blocks | 46.2 µs | **17.6 µs** | 0.38× | 34% | 61.8 µs | **37.1 µs** |
| conv-1D simple | ComplexF64 200⋆200 → n=400 | 28.8 µs | **13.1 µs** | 0.46× | 26% | 305.0 µs | **51.8 µs** |
| conv-1D overlapsave | ComplexF64 200⋆200 → nfft=400, 2 blocks | 34.4 µs | **19.2 µs** | 0.56× | 18% | 47.2 µs | **30.2 µs** |
| conv-2D simple | Float64 10×20 ⋆ 20×10 → 30×30 | 71.8 µs | **80.6 µs** | 1.12× | 20% | 439.8 µs | **60.85 ms** |
| conv-2D overlapsave | Float64 10×20 ⋆ 20×10 → nfft=40, 2 blocks | 103.4 µs | **113.6 µs** | 1.10× | 11% | 1.38 ms | **181.9 µs** |
| conv-2D simple | ComplexF64 10×20 ⋆ 20×10 → 30×30 | 91.4 µs | **143.8 µs** | 1.57× | 7% | 913.3 µs | **120.58 ms** |
| conv-2D overlapsave | ComplexF64 10×20 ⋆ 20×10 → nfft=40, 2 blocks | 163.0 µs | **131.3 µs** | 0.81× | 2% | 1.19 ms | **173.0 µs** |
| conv-2D simple | Float64 10×20 ⋆ 210×200 → 224×224 | 1.21 ms | **1.38 ms** | 1.14× | 1% | 2.13 ms | **3.11 ms** |
| conv-2D overlapsave | Float64 10×20 ⋆ 210×200 → nfft=128, 6 blocks | 932.8 µs | **1.73 ms** | 1.86× | 1% | 1.02 ms | **1.82 ms** |
| conv-2D simple | ComplexF64 10×20 ⋆ 210×200 → 224×224 | 2.46 ms | **2.63 ms** | 1.07× | 0% | 4.06 ms | **2.69 ms** |
| conv-2D overlapsave | ComplexF64 10×20 ⋆ 210×200 → nfft=128, 6 blocks | 1.74 ms | **3.32 ms** | 1.90× | 0% | 1.84 ms | **3.40 ms** |
| conv-2D simple | Float64 190×200 ⋆ 20×10 → 210×210 | 1.53 ms | **1.75 ms** | 1.15× | 1% | 1.73 ms | **2.20 ms** |
| conv-2D overlapsave | Float64 190×200 ⋆ 20×10 → nfft=128, 4 blocks | 648.1 µs | **1.21 ms** | 1.87× | 1% | 687.0 µs | **1.27 ms** |
| conv-2D simple | ComplexF64 190×200 ⋆ 20×10 → 210×210 | 3.01 ms | **3.14 ms** | 1.04× | 0% | 3.91 ms | **3.26 ms** |
| conv-2D overlapsave | ComplexF64 190×200 ⋆ 20×10 → nfft=128, 4 blocks | 1.21 ms | **2.30 ms** | 1.91× | 0% | 1.74 ms | **2.34 ms** |
| conv-2D simple | Float64 190×200 ⋆ 210×200 → 400×400 | 4.39 ms | **5.34 ms** | 1.22× | 0% | 4.95 ms | **7.13 ms** |
| conv-2D overlapsave | Float64 190×200 ⋆ 210×200 → nfft=420, 4 blocks | 18.41 ms | **21.61 ms** | 1.17× | 0% | 20.64 ms | **21.58 ms** |
| conv-2D simple | ComplexF64 190×200 ⋆ 210×200 → 400×400 | 8.08 ms | **9.95 ms** | 1.23× | 0% | 8.85 ms | **10.03 ms** |
| conv-2D overlapsave | ComplexF64 190×200 ⋆ 210×200 → nfft=420, 4 blocks | 35.22 ms | **40.12 ms** | 1.14× | 0% | 36.85 ms | **40.56 ms** |
| conv-ND small | Float64 5 ⋆ 5 → 9 | 4.6 µs | **2.2 µs** | 0.47× | 74% | 22.8 µs | **49.8 µs** |
| conv-ND small | Float64 5×5 ⋆ 5×5 → 9×9 | 15.6 µs | **16.3 µs** | 1.04× | 66% | 61.6 µs | **52.8 µs** |
| conv-ND small | Float64 5×5×5 ⋆ 5×5×5 → 9×9×9 | 50.6 µs | **71.4 µs** | 1.41× | 17% | 160.3 µs | **190.54 ms** |
| conv-ND small | Float64 3×3×6 ⋆ 2×2×2 → 4×4×7 | 38.9 µs | **21.4 µs** | 0.55× | 48% | 67.2 µs | **41.7 µs** |
| conv-ND small | Float64 2×2×2×2×2×2 ⋆ 1×1×1×1×1×1 → 2×2×2×2×2×2 | 108.9 µs | **47.7 µs** | 0.44× | 30% | 628.7 µs | **2.92 s** |
| conv-ND small | Float64 4×7×1 ⋆ 3×3×3 → 6×9×3 | 39.9 µs | **26.2 µs** | 0.66× | 42% | 77.9 µs | **44.2 µs** |
| os-test overlapsave | Float32 128^1 ⋆ 12^1, nfft=64, 3 blocks | 11.8 µs | **2.7 µs** | 0.23× | 49% | 259.4 µs | **25.2 µs** |
| os-test reference | Float32 128^1 ⋆ 12^1 → 140^1 | 40.4 µs | **4.8 µs** | 0.12× | 48% | 538.0 µs | **44.66 ms** |
| os-test overlapsave | Float32 128^1 ⋆ 128^1, nfft=256, 2 blocks | 43.1 µs | **6.3 µs** | 0.15× | 47% | 787.0 µs | **53.9 µs** |
| os-test reference | Float32 128^1 ⋆ 128^1 → 256^1 | 42.2 µs | **6.0 µs** | 0.14× | 55% | 52.8 µs | **20.6 µs** |
| os-test overlapsave | Float64 128^1 ⋆ 12^1, nfft=64, 3 blocks | 11.8 µs | **4.0 µs** | 0.34× | 47% | 37.4 µs | **16.8 µs** |
| os-test reference | Float64 128^1 ⋆ 12^1 → 140^1 | 41.4 µs | **6.4 µs** | 0.15× | 48% | 420.2 µs | **30.2 µs** |
| os-test overlapsave | Float64 128^1 ⋆ 128^1, nfft=256, 2 blocks | 30.8 µs | **11.6 µs** | 0.38× | 53% | 769.4 µs | **24.9 µs** |
| os-test reference | Float64 128^1 ⋆ 128^1 → 256^1 | 30.0 µs | **11.4 µs** | 0.38× | 57% | 41.6 µs | **26.1 µs** |
| os-test overlapsave | ComplexF64 128^1 ⋆ 12^1, nfft=64, 3 blocks | 22.6 µs | **6.1 µs** | 0.27× | 44% | 43.5 µs | **31.9 µs** |
| os-test reference | ComplexF64 128^1 ⋆ 12^1 → 140^1 | 23.2 µs | **7.2 µs** | 0.31× | 37% | 286.7 µs | **41.2 µs** |
| os-test overlapsave | ComplexF64 128^1 ⋆ 128^1, nfft=256, 2 blocks | 28.2 µs | **14.8 µs** | 0.53× | 36% | 261.9 µs | **35.0 µs** |
| os-test reference | ComplexF64 128^1 ⋆ 128^1 → 256^1 | 25.0 µs | **11.0 µs** | 0.44× | 47% | 40.8 µs | **27.7 µs** |
| os-test overlapsave | Float32 128^2 ⋆ 12^2, nfft=64, 9 blocks | 263.4 µs | **422.2 µs** | 1.60× | 2% | 361.3 µs | **567.4 µs** |
| os-test reference | Float32 128^2 ⋆ 12^2 → 140^2 | 408.8 µs | **555.6 µs** | 1.36× | 2% | 671.4 µs | **52.69 ms** |
| os-test overlapsave | Float32 128^2 ⋆ 128^2, nfft=256, 4 blocks | 3.72 ms | **3.10 ms** | 0.83× | 0% | 4.17 ms | **3.18 ms** |
| os-test reference | Float32 128^2 ⋆ 128^2 → 256^2 | 1.30 ms | **1.06 ms** | 0.82× | 1% | 1.38 ms | **1.12 ms** |
| os-test overlapsave | Float64 128^2 ⋆ 12^2, nfft=64, 9 blocks | 299.8 µs | **652.0 µs** | 2.18× | 2% | 396.0 µs | **681.0 µs** |
| os-test reference | Float64 128^2 ⋆ 12^2 → 140^2 | 512.1 µs | **700.8 µs** | 1.37× | 2% | 785.0 µs | **774.8 µs** |
| os-test overlapsave | Float64 128^2 ⋆ 128^2, nfft=256, 4 blocks | 4.59 ms | **5.22 ms** | 1.14× | 0% | 5.03 ms | **5.25 ms** |
| os-test reference | Float64 128^2 ⋆ 128^2 → 256^2 | 1.56 ms | **1.76 ms** | 1.13× | 1% | 1.63 ms | **4.11 ms** |
| os-test overlapsave | ComplexF64 128^2 ⋆ 12^2, nfft=64, 9 blocks | 508.5 µs | **1.21 ms** | 2.37× | 0% | 591.8 µs | **1.25 ms** |
| os-test reference | ComplexF64 128^2 ⋆ 12^2 → 140^2 | 971.2 µs | **1.21 ms** | 1.25× | 0% | 1.66 ms | **1.31 ms** |
| os-test overlapsave | ComplexF64 128^2 ⋆ 128^2, nfft=256, 4 blocks | 11.40 ms | **11.80 ms** | 1.04× | 0% | 12.21 ms | **12.11 ms** |
| os-test reference | ComplexF64 128^2 ⋆ 128^2 → 256^2 | 4.00 ms | **3.95 ms** | 0.99× | 0% | 4.07 ms | **4.03 ms** |
| os-test overlapsave | Float32 128^3 ⋆ 12^3, nfft=64, 27 blocks | 80.61 ms | **129.02 ms** | 1.60× | 0% | 81.03 ms | **129.19 ms** |
| os-test reference | Float32 128^3 ⋆ 12^3 → 140^3 | 87.12 ms | **120.79 ms** | 1.39× | 0% | 90.51 ms | **229.76 ms** |
| os-test overlapsave | Float32 128^3 ⋆ 128^3, nfft=256, 8 blocks | 4.17 s | **3.14 s** | 0.75× | 0% | 4.23 s | **3.14 s** |
| os-test reference | Float32 128^3 ⋆ 128^3 → 256^3 | 792.07 ms | **662.43 ms** | 0.84× | 0% | 1.07 s | **647.59 ms** |
| os-test overlapsave | Float64 128^3 ⋆ 12^3, nfft=64, 27 blocks | 107.06 ms | **204.98 ms** | 1.91× | 0% | 108.51 ms | **206.63 ms** |
| os-test reference | Float64 128^3 ⋆ 12^3 → 140^3 | 128.75 ms | **201.35 ms** | 1.56× | 0% | 173.19 ms | **194.76 ms** |
| os-test overlapsave | Float64 128^3 ⋆ 128^3, nfft=256, 8 blocks | 5.19 s | **5.16 s** | 0.99× | 0% | 5.31 s | **5.37 s** |
| os-test reference | Float64 128^3 ⋆ 128^3 → 256^3 | 1.30 s | **1.09 s** | 0.84× | 0% | 1.44 s | **1.30 s** |
| os-test overlapsave | ComplexF64 128^3 ⋆ 12^3, nfft=64, 27 blocks | 193.15 ms | **413.25 ms** | 2.14× | 0% | 194.66 ms | **509.96 ms** |
| os-test reference | ComplexF64 128^3 ⋆ 12^3 → 140^3 | 220.21 ms | **303.59 ms** | 1.38× | 0% | 221.24 ms | **319.14 ms** |
| os-test overlapsave | ComplexF64 128^3 ⋆ 128^3, nfft=256, 8 blocks | 10.84 s | **11.20 s** | 1.03× | 0% | 10.86 s | **11.48 s** |
| os-test reference | ComplexF64 128^3 ⋆ 128^3 → 256^3 | 2.02 s | **2.08 s** | 1.03× | 0% | 2.03 s | **2.09 s** |
| os-test adversarial | Float64 128 ⋆ 12, nfft=256, 1 blocks | 28.6 µs | **9.8 µs** | 0.34× | 78% | 56.1 µs | **23.3 µs** |
| os-test adversarial | Float64 128 ⋆ 13, nfft=32, 7 blocks | 12.1 µs | **3.6 µs** | 0.30× | 36% | 132.1 µs | **16.4 µs** |
| os-test adversarial | Float64 128 ⋆ 12, nfft=32, 7 blocks | 12.0 µs | **3.6 µs** | 0.30× | 34% | 19.4 µs | **13.3 µs** |
| os-test adversarial | Float64 25 ⋆ 4, nfft=16, 3 blocks | 4.8 µs | **1.8 µs** | 0.38× | 60% | 28.5 µs | **12.0 µs** |

## Totals per group

| group | FFTW steady | FFTA steady | FFTA / FFTW | FFTW first calls | FFTA first calls | first-call ratio |
|:--|--:|--:|--:|--:|--:|--:|
| conv-1D simple | 251.1 µs | **72.4 µs** | 0.29× | 2.62 ms | **159.63 ms** | 60.96× |
| conv-1D overlapsave | 167.4 µs | **68.6 µs** | 0.41× | 702.7 µs | **207.5 µs** | 0.30× |
| conv-2D simple | 20.85 ms | **24.41 ms** | 1.17× | 26.98 ms | **209.85 ms** | 7.78× |
| conv-2D overlapsave | 58.43 ms | **70.54 ms** | 1.21× | 65.36 ms | **71.32 ms** | 1.09× |
| conv-ND small | 258.6 µs | **185.1 µs** | 0.72× | 1.02 ms | **3.11 s** | 3050.84× |
| os-test overlapsave | 20.59 s | **20.27 s** | 0.98× | 20.81 s | **20.85 s** | 1.00× |
| os-test reference | 4.56 s | **4.47 s** | 0.98× | 5.04 s | **4.89 s** | 0.97× |
| os-test adversarial | 57.5 µs | **18.8 µs** | 0.33× | 236.1 µs | **65.0 µs** | 0.28× |
| **all** | 25.23 s | **24.84 s** | 0.98× | 25.94 s | **29.29 s** | 1.13× |

## N-d transforms, one dimension at a time (`nd_stages.jl`)

| shape | pass | FFTW | FFTA | FFTA / FFTW | FFTA 1-D × pencils | pencil overhead |
|:--|:--|--:|--:|--:|--:|--:|
| ComplexF64 140×140 | dim 1 (140 pencils of 140) | 110.8 µs | **168.9 µs** | 1.52× | 151.2 µs | 1.12× |
| ComplexF64 140×140 | dim 2 (140 pencils of 140) | 122.0 µs | **183.5 µs** | 1.50× | 151.2 µs | 1.21× |
| ComplexF64 140×140 | **all dims** | 244.6 µs | **363.4 µs** | 1.49× | | |
| Float64 140×140 | dim 1 (140 pencils of 140) | 56.8 µs | **110.0 µs** | 1.94× | 106.4 µs | 1.03× |
| Float64 140×140 | **all dims** | 122.3 µs | **198.5 µs** | 1.62× | | |
| ComplexF64 256×256 | dim 1 (256 pencils of 256) | 372.1 µs | **441.1 µs** | 1.19× | 368.6 µs | 1.20× |
| ComplexF64 256×256 | dim 2 (256 pencils of 256) | 860.2 µs | **741.6 µs** | 0.86× | 368.6 µs | 2.01× |
| ComplexF64 256×256 | **all dims** | 1.14 ms | **1.18 ms** | 1.03× | | |
| Float64 256×256 | dim 1 (256 pencils of 256) | 186.4 µs | **252.8 µs** | 1.36× | 235.5 µs | 1.07× |
| Float64 256×256 | **all dims** | 413.4 µs | **487.5 µs** | 1.18× | | |
| ComplexF64 140×140×140 | dim 1 (19600 pencils of 140) | 20.87 ms | **29.18 ms** | 1.40× | 21.17 ms | 1.38× |
| ComplexF64 140×140×140 | dim 2 (19600 pencils of 140) | 26.47 ms | **33.96 ms** | 1.28× | 21.17 ms | 1.60× |
| ComplexF64 140×140×140 | dim 3 (19600 pencils of 140) | 27.51 ms | **38.61 ms** | 1.40× | 21.17 ms | 1.82× |
| ComplexF64 140×140×140 | **all dims** | 66.41 ms | **91.10 ms** | 1.37× | | |
| Float64 140×140×140 | dim 1 (19600 pencils of 140) | 8.59 ms | **17.99 ms** | 2.10× | 14.90 ms | 1.21× |
| Float64 140×140×140 | **all dims** | 32.53 ms | **51.55 ms** | 1.58× | | |
| ComplexF64 64×64×64 | dim 1 (4096 pencils of 64) | 654.2 µs | **1.76 ms** | 2.69× | 1.47 ms | 1.19× |
| ComplexF64 64×64×64 | dim 2 (4096 pencils of 64) | 1.07 ms | **2.20 ms** | 2.05× | 1.47 ms | 1.49× |
| ComplexF64 64×64×64 | dim 3 (4096 pencils of 64) | 1.29 ms | **2.64 ms** | 2.05× | 1.47 ms | 1.79× |
| ComplexF64 64×64×64 | **all dims** | 2.51 ms | **6.49 ms** | 2.58× | | |
| Float64 64×64×64 | dim 1 (4096 pencils of 64) | 444.6 µs | **1.16 ms** | 2.62× | 983.0 µs | 1.18× |
| Float64 64×64×64 | **all dims** | 1.36 ms | **3.19 ms** | 2.34× | | |
