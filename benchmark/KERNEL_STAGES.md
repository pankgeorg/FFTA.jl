# Kernel stage breakdown (ComplexF64, single thread, neoverse-n1, Julia 1.12.6)

Stage times are minimum times of the kernel re-run with the other stages removed; `copy` is one `copyto!` of the array (the memory-traffic floor of a single pass).

Branch `integration/all` @ 901692b (before the SIMD-pass / leaves-first / codelet branches). Each block re-runs the kernel with stages removed, so the stage rows add up to the total; the last column relates each stage to one `copyto!` of the array (the memory-traffic floor of a single pass). Hardware counters were not available on this host.
### n = 1024 = 2^10 (radix-4 recursion, 2 butterfly levels, 16 leaves of 64 points)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 346 ns |
|:--|--:|--:|--:|
| FFTA total | 9.52 µs | 100% | 27.5× copy |
| leaves: 16 × 64-point codelet, input stride 16 | 5.32 µs | 56% | 15.4× copy |
| (same codelets on contiguous input) | 5.28 µs | 55% | strided-load penalty 1.01× |
| butterfly pass, blocks of 1024 (stride 256 between the 4 legs) | 2.11 µs | 22% | 6.1× copy; memory-only version of the pass 620 ns |
| butterfly pass, blocks of 256 (stride 64 between the 4 legs) | 2.14 µs | 22% | 6.2× copy; memory-only version of the pass 631 ns |
| leaves + passes (sum of the rows) | 9.56 µs | 100% | recursion/dispatch overhead = total − sum = -43 ns |
| FFTW ESTIMATE / MEASURE | 6.32 µs / 5.28 µs | 66% / 55% | 18.2× / 15.2× copy |

### n = 16384 = 2^14 (radix-4 recursion, 4 butterfly levels, 256 leaves of 64 points)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 6.04 µs |
|:--|--:|--:|--:|
| FFTA total | 244.68 µs | 100% | 40.5× copy |
| leaves: 256 × 64-point codelet, input stride 256 | 97.64 µs | 40% | 16.2× copy |
| (same codelets on contiguous input) | 86.64 µs | 35% | strided-load penalty 1.13× |
| butterfly pass, blocks of 16384 (stride 4096 between the 4 legs) | 34.16 µs | 14% | 5.7× copy; memory-only version of the pass 14.08 µs |
| butterfly pass, blocks of 4096 (stride 1024 between the 4 legs) | 34.40 µs | 14% | 5.7× copy; memory-only version of the pass 14.14 µs |
| butterfly pass, blocks of 1024 (stride 256 between the 4 legs) | 33.84 µs | 14% | 5.6× copy; memory-only version of the pass 12.52 µs |
| butterfly pass, blocks of 256 (stride 64 between the 4 legs) | 34.72 µs | 14% | 5.7× copy; memory-only version of the pass 12.86 µs |
| leaves + passes (sum of the rows) | 234.76 µs | 96% | recursion/dispatch overhead = total − sum = 9.92 µs |
| FFTW ESTIMATE / MEASURE | 176.64 µs / 157.64 µs | 72% / 64% | 29.2× / 26.1× copy |

### n = 262144 = 2^18 (radix-4 recursion, 6 butterfly levels, 4096 leaves of 64 points)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 334.44 µs |
|:--|--:|--:|--:|
| FFTA total | 6.733 ms | 100% | 20.1× copy |
| leaves: 4096 × 64-point codelet, input stride 4096 | 3.198 ms | 47% | 9.6× copy |
| (same codelets on contiguous input) | 1.465 ms | 22% | strided-load penalty 2.18× |
| butterfly pass, blocks of 262144 (stride 65536 between the 4 legs) | 560.61 µs | 8% | 1.7× copy; memory-only version of the pass 241.20 µs |
| butterfly pass, blocks of 65536 (stride 16384 between the 4 legs) | 557.29 µs | 8% | 1.7× copy; memory-only version of the pass 237.44 µs |
| butterfly pass, blocks of 16384 (stride 4096 between the 4 legs) | 553.53 µs | 8% | 1.7× copy; memory-only version of the pass 232.48 µs |
| butterfly pass, blocks of 4096 (stride 1024 between the 4 legs) | 556.89 µs | 8% | 1.7× copy; memory-only version of the pass 233.80 µs |
| butterfly pass, blocks of 1024 (stride 256 between the 4 legs) | 561.65 µs | 8% | 1.7× copy; memory-only version of the pass 219.52 µs |
| butterfly pass, blocks of 256 (stride 64 between the 4 legs) | 600.41 µs | 9% | 1.8× copy; memory-only version of the pass 245.24 µs |
| leaves + passes (sum of the rows) | 6.588 ms | 98% | recursion/dispatch overhead = total − sum = 145.52 µs |
| FFTW ESTIMATE / MEASURE | 8.897 ms / 4.631 ms | 132% / 69% | 26.6× / 13.8× copy |

### n = 4194304 = 2^22 (radix-4 recursion, 8 butterfly levels, 65536 leaves of 64 points)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 6.120 ms |
|:--|--:|--:|--:|
| FFTA total | 210.500 ms | 100% | 34.4× copy |
| leaves: 65536 × 64-point codelet, input stride 65536 | 124.993 ms | 59% | 20.4× copy |
| (same codelets on contiguous input) | 33.230 ms | 16% | strided-load penalty 3.76× |
| butterfly pass, blocks of 4194304 (stride 1048576 between the 4 legs) | 9.916 ms | 5% | 1.6× copy; memory-only version of the pass 7.003 ms |
| butterfly pass, blocks of 1048576 (stride 262144 between the 4 legs) | 9.250 ms | 4% | 1.5× copy; memory-only version of the pass 4.768 ms |
| butterfly pass, blocks of 262144 (stride 65536 between the 4 legs) | 9.076 ms | 4% | 1.5× copy; memory-only version of the pass 4.336 ms |
| butterfly pass, blocks of 65536 (stride 16384 between the 4 legs) | 9.051 ms | 4% | 1.5× copy; memory-only version of the pass 4.255 ms |
| butterfly pass, blocks of 16384 (stride 4096 between the 4 legs) | 9.064 ms | 4% | 1.5× copy; memory-only version of the pass 4.171 ms |
| butterfly pass, blocks of 4096 (stride 1024 between the 4 legs) | 9.362 ms | 4% | 1.5× copy; memory-only version of the pass 4.570 ms |
| butterfly pass, blocks of 1024 (stride 256 between the 4 legs) | 10.913 ms | 5% | 1.8× copy; memory-only version of the pass 5.828 ms |
| butterfly pass, blocks of 256 (stride 64 between the 4 legs) | 11.518 ms | 5% | 1.9× copy; memory-only version of the pass 6.324 ms |
| leaves + passes (sum of the rows) | 203.144 ms | 97% | recursion/dispatch overhead = total − sum = 7.357 ms |
| FFTW ESTIMATE / MEASURE | 242.198 ms / 104.928 ms | 115% / 50% | 39.6× / 17.1× copy |
### n = 1000 = 8 × 125 (composite step: right = 125 COMPOSITE_FFT, left = 8 POW2RADIX4_FFT)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 338 ns |
|:--|--:|--:|--:|
| FFTA total | 35.32 µs | 100% | 104.4× copy |
| 8 right sub-transforms of 125 (input stride 8) | 30.80 µs | 87% | 91.0× copy |
| twiddle multiply pass | 1.27 µs | 4% | 3.7× copy |
| 125 left sub-transforms of 8 (output stride 125) | 3.09 µs | 9% | 9.1× copy |
| sum of the rows | 35.15 µs | 100% | overhead = 169 ns |
| FFTW ESTIMATE / MEASURE | 10.28 µs / 7.36 µs | 29% / 21% | 30.4× / 21.7× copy |

### n = 46305 = 27 × 1715 (composite step: right = 1715 COMPOSITE_FFT, left = 27 POW3_FFT)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 27.56 µs |
|:--|--:|--:|--:|
| FFTA total | 3.326 ms | 100% | 120.7× copy |
| 27 right sub-transforms of 1715 (input stride 27) | 2.521 ms | 76% | 91.5× copy |
| twiddle multiply pass | 68.76 µs | 2% | 2.5× copy |
| 1715 left sub-transforms of 27 (output stride 1715) | 693.69 µs | 21% | 25.2× copy |
| sum of the rows | 3.284 ms | 99% | overhead = 42.32 µs |
| FFTW ESTIMATE / MEASURE | 1.052 ms / 865.49 µs | 32% / 26% | 38.2× / 31.4× copy |

### n = 1000000 = 64 × 15625 (composite step: right = 15625 COMPOSITE_FFT, left = 64 POW2RADIX4_FFT)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 1.274 ms |
|:--|--:|--:|--:|
| FFTA total | 100.257 ms | 100% | 78.7× copy |
| 64 right sub-transforms of 15625 (input stride 64) | 83.168 ms | 83% | 65.3× copy |
| twiddle multiply pass | 2.005 ms | 2% | 1.6× copy |
| 15625 left sub-transforms of 64 (output stride 15625) | 11.154 ms | 11% | 8.8× copy |
| sum of the rows | 96.327 ms | 96% | overhead = 3.930 ms |
| FFTW ESTIMATE / MEASURE | 55.945 ms / 24.771 ms | 56% / 25% | 43.9× / 19.4× copy |

### n = 65537 (Bluestein, padded to 262144)

| stage | time | share of FFTA total | time of one `copyto!` of the array: 60.56 µs |
|:--|--:|--:|--:|
| FFTA total | 15.552 ms | 100% | 256.8× copy |
| chirp multiply + zero pad | 191.36 µs | 1% | |
| forward FFT of 262144 | 6.390 ms | 41% | |
| pointwise product | 540.45 µs | 3% | |
| second FFT of 262144 | 6.401 ms | 41% | |
| output chirp multiply | 131.68 µs | 1% | |
| FFTW ESTIMATE / MEASURE | 3.362 ms / 2.660 ms | 22% / 17% | |
