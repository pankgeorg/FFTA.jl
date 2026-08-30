| case (4 threads) | FFTW ESTIMATE | FFTW MEASURE | FFTA (all) | FFTA / MEASURE |
|:--|--:|--:|--:|--:|
| fft ComplexF64 1024 dims=1 | 14.08 µs | 13.32 µs | **6.97 µs** | 0.52× |
| fft ComplexF64 65536 dims=1 | 354.16 µs | 303.96 µs | **1.062 ms** | 3.49× |
| fft ComplexF64 1048576 dims=1 | 11.399 ms | 5.510 ms | **5.451 ms** | 0.99× |
| fft ComplexF32 65536 dims=1 | 199.88 µs | 176.24 µs | **587.77 µs** | 3.33× |
| fft ComplexF32 1048576 dims=1 | 6.585 ms | 3.827 ms | **3.194 ms** | 0.83× |
| fft ComplexF64 1000 dims=1 | 14.60 µs | 14.60 µs | **14.78 µs** | 1.01× |
| fft ComplexF64 1000000 dims=1 | 10.061 ms | 7.473 ms | **49.644 ms** | 6.64× |
| fft ComplexF64 65537 dims=1 | 2.198 ms | 2.136 ms | **9.101 ms** | 4.26× |
| fft ComplexF64 12297 dims=1 | 368.04 µs | 312.64 µs | **1.345 ms** | 4.30× |
| rfft Float64 4096 dims=1 | 24.16 µs | 24.44 µs | **19.80 µs** | 0.81× |
| rfft Float64 65536 dims=1 | 388.80 µs | 283.72 µs | **537.93 µs** | 1.90× |
| rfft Float64 1048576 dims=1 | 6.353 ms | 4.746 ms | **5.368 ms** | 1.13× |
| rfft Float32 65536 dims=1 | 264.80 µs | 238.20 µs | **302.80 µs** | 1.27× |
| fft ComplexF64 256×256 dims=(1, 2) | 382.41 µs | 255.56 µs | **430.97 µs** | 1.69× |
| fft ComplexF64 1024×1024 dims=(1, 2) | 8.207 ms | 4.699 ms | **8.877 ms** | 1.89× |
| rfft Float64 1024×1024 dims=(1, 2) | 2.389 ms | 2.247 ms | **2.609 ms** | 1.16× |
| fft ComplexF64 64×64×64 dims=(1, 2, 3) | 773.81 µs | 775.33 µs | **2.214 ms** | 2.86× |
| fft ComplexF64 4096×64 dims=1 | 577.65 µs | 546.13 µs | **642.61 µs** | 1.18× |
| rfft Float64 4096×64 dims=1 | 281.96 µs | 282.84 µs | **330.48 µs** | 1.17× |
| fft ComplexF64 64×4096 dims=2 | 1.462 ms | 1.431 ms | **1.704 ms** | 1.19× |
| rfft Float64 1024×1024 dims=1 | 968.41 µs | 974.57 µs | **1.167 ms** | 1.20× |
