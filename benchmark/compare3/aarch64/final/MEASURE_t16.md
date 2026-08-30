| case (16 threads) | FFTW ESTIMATE | FFTW MEASURE | FFTA (all) | FFTA / MEASURE |
|:--|--:|--:|--:|--:|
| fft ComplexF64 1024 dims=1 | 30.32 µs | 28.76 µs | **6.97 µs** | 0.24× |
| fft ComplexF64 65536 dims=1 | 142.60 µs | 116.60 µs | **1.047 ms** | 8.98× |
| fft ComplexF64 1048576 dims=1 | 3.033 ms | 1.785 ms | **1.841 ms** | 1.03× |
| fft ComplexF32 65536 dims=1 | 83.40 µs | 86.84 µs | **639.85 µs** | 7.37× |
| fft ComplexF32 1048576 dims=1 | 2.171 ms | 1.259 ms | **1.125 ms** | 0.89× |
| fft ComplexF64 1000 dims=1 | 44.20 µs | 33.64 µs | **14.78 µs** | 0.44× |
| fft ComplexF64 1000000 dims=1 | 2.925 ms | 2.485 ms | **47.496 ms** | 19.12× |
| fft ComplexF64 65537 dims=1 | 1.532 ms | 1.506 ms | **9.104 ms** | 6.05× |
| fft ComplexF64 12297 dims=1 | 263.92 µs | 215.40 µs | **1.330 ms** | 6.18× |
| rfft Float64 4096 dims=1 | 14.54 µs | 31.44 µs | **19.82 µs** | 0.63× |
| rfft Float64 65536 dims=1 | 345.92 µs | 173.56 µs | **526.37 µs** | 3.03× |
| rfft Float64 1048576 dims=1 | 5.328 ms | 2.707 ms | **3.539 ms** | 1.31× |
| rfft Float32 65536 dims=1 | 218.16 µs | 186.32 µs | **299.56 µs** | 1.61× |
| fft ComplexF64 256×256 dims=(1, 2) | 2.916 ms | 90.56 µs | **156.36 µs** | 1.73× |
| fft ComplexF64 1024×1024 dims=(1, 2) | 2.282 ms | 1.650 ms | **3.375 ms** | 2.05× |
| rfft Float64 1024×1024 dims=(1, 2) | 2.615 ms | 719.41 µs | **779.89 µs** | 1.08× |
| fft ComplexF64 64×64×64 dims=(1, 2, 3) | 260.52 µs | 263.40 µs | **797.97 µs** | 3.03× |
| fft ComplexF64 4096×64 dims=1 | 204.12 µs | 191.20 µs | **150.68 µs** | 0.79× |
| rfft Float64 4096×64 dims=1 | 102.76 µs | 102.56 µs | **81.44 µs** | 0.79× |
| fft ComplexF64 64×4096 dims=2 | 411.53 µs | 407.77 µs | **489.01 µs** | 1.20× |
| rfft Float64 1024×1024 dims=1 | 294.48 µs | 290.36 µs | **288.96 µs** | 1.00× |
