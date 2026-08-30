| case (1 thread) | FFTW ESTIMATE | FFTW MEASURE | FFTA (all) | FFTA / MEASURE |
|:--|--:|--:|--:|--:|
| fft ComplexF64 1024 dims=1 | 6.30 µs | 5.30 µs | **6.95 µs** | 1.31× |
| fft ComplexF64 65536 dims=1 | 1.138 ms | 844.05 µs | **1.015 ms** | 1.20× |
| fft ComplexF64 1048576 dims=1 | 47.243 ms | 21.186 ms | **19.172 ms** | 0.90× |
| fft ComplexF32 65536 dims=1 | 576.65 µs | 463.93 µs | **563.93 µs** | 1.22× |
| fft ComplexF32 1048576 dims=1 | 25.148 ms | 14.385 ms | **11.271 ms** | 0.78× |
| fft ComplexF64 1000 dims=1 | 10.26 µs | 7.37 µs | **14.66 µs** | 1.99× |
| fft ComplexF64 1000000 dims=1 | 46.157 ms | 24.658 ms | **47.349 ms** | 1.92× |
| fft ComplexF64 65537 dims=1 | 3.358 ms | 2.679 ms | **8.967 ms** | 3.35× |
| fft ComplexF64 12297 dims=1 | 790.81 µs | 617.29 µs | **1.305 ms** | 2.11× |
| rfft Float64 4096 dims=1 | 14.54 µs | 14.48 µs | **19.76 µs** | 1.36× |
| rfft Float64 65536 dims=1 | 475.21 µs | 414.61 µs | **535.53 µs** | 1.29× |
| rfft Float64 1048576 dims=1 | 12.448 ms | 9.858 ms | **11.380 ms** | 1.15× |
| rfft Float32 65536 dims=1 | 246.76 µs | 211.00 µs | **305.04 µs** | 1.45× |
| fft ComplexF64 256×256 dims=(1, 2) | 1.281 ms | 764.05 µs | **1.227 ms** | 1.61× |
| fft ComplexF64 1024×1024 dims=(1, 2) | 31.142 ms | 17.783 ms | **30.049 ms** | 1.69× |
| rfft Float64 1024×1024 dims=(1, 2) | 8.935 ms | 7.862 ms | **9.620 ms** | 1.22× |
| fft ComplexF64 64×64×64 dims=(1, 2, 3) | 2.525 ms | 2.452 ms | **6.754 ms** | 2.75× |
| fft ComplexF64 4096×64 dims=1 | 2.182 ms | 2.046 ms | **2.614 ms** | 1.28× |
| rfft Float64 4096×64 dims=1 | 1.068 ms | 1.064 ms | **1.325 ms** | 1.24× |
| fft ComplexF64 64×4096 dims=2 | 5.692 ms | 5.170 ms | **6.366 ms** | 1.23× |
| rfft Float64 1024×1024 dims=1 | 3.540 ms | 3.533 ms | **4.604 ms** | 1.30× |
