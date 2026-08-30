| case (16 threads) | FFTW ESTIMATE | FFTW MEASURE | FFTA (all) | FFTA / MEASURE |
|:--|--:|--:|--:|--:|
| fft ComplexF64 1024 dims=1 | 29.20 µs | 23.32 µs | **6.95 µs** | 0.30× |
| fft ComplexF64 65536 dims=1 | 135.20 µs | 106.80 µs | **1.089 ms** | 10.19× |
| fft ComplexF64 1048576 dims=1 | 3.265 ms | 1.889 ms | **19.815 ms** | 10.49× |
| fft ComplexF32 65536 dims=1 | 77.32 µs | 85.48 µs | **583.45 µs** | 6.83× |
| fft ComplexF32 1048576 dims=1 | 1.911 ms | 1.406 ms | **11.219 ms** | 7.98× |
| fft ComplexF64 1000 dims=1 | 42.68 µs | 33.68 µs | **14.70 µs** | 0.44× |
| fft ComplexF64 1000000 dims=1 | 3.043 ms | 2.593 ms | **48.243 ms** | 18.60× |
| fft ComplexF64 65537 dims=1 | 1.540 ms | 1.611 ms | **9.102 ms** | 5.65× |
| fft ComplexF64 12297 dims=1 | 261.72 µs | 213.96 µs | **1.321 ms** | 6.17× |
| rfft Float64 4096 dims=1 | 14.52 µs | 32.08 µs | **19.80 µs** | 0.62× |
| rfft Float64 65536 dims=1 | 336.80 µs | 165.88 µs | **513.89 µs** | 3.10× |
| rfft Float64 1048576 dims=1 | 5.139 ms | 2.427 ms | **11.567 ms** | 4.77× |
| rfft Float32 65536 dims=1 | 235.88 µs | 215.84 µs | **302.52 µs** | 1.40× |
| fft ComplexF64 256×256 dims=(1, 2) | 3.116 ms | 88.28 µs | **158.72 µs** | 1.80× |
| fft ComplexF64 1024×1024 dims=(1, 2) | 2.309 ms | 1.638 ms | **3.317 ms** | 2.02× |
| rfft Float64 1024×1024 dims=(1, 2) | 2.649 ms | 715.05 µs | **771.21 µs** | 1.08× |
| fft ComplexF64 64×64×64 dims=(1, 2, 3) | 252.68 µs | 252.72 µs | **813.57 µs** | 3.22× |
| fft ComplexF64 4096×64 dims=1 | 196.28 µs | 185.60 µs | **149.20 µs** | 0.80× |
| rfft Float64 4096×64 dims=1 | 99.96 µs | 105.48 µs | **81.80 µs** | 0.78× |
| fft ComplexF64 64×4096 dims=2 | 413.01 µs | 410.72 µs | **483.93 µs** | 1.18× |
| rfft Float64 1024×1024 dims=1 | 274.04 µs | 318.36 µs | **278.72 µs** | 0.88× |
