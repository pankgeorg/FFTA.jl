### simd vs int (speedup, >1 = faster) and simd/fftw (ratio, lower = better)

| group | type | speedup vs int | simd/fftw | cases |
|:--|:--|--:|--:|--:|
| pow2 | Float64 | **1.12×** | 1.65× | 40 |
| pow2 | Float32 | **1.46×** | 1.45× | 40 |
| smooth | Float64 | **0.98×** | 4.29× | 48 |
| smooth | Float32 | **1.00×** | 6.62× | 48 |
| prime | Float64 | **1.11×** | 1.21× | 48 |
| prime | Float32 | **1.44×** | 1.14× | 48 |
| awkward | Float64 | **1.02×** | 1.69× | 58 |
| awkward | Float32 | **1.50×** | 1.76× | 58 |
| 2d | Float64 | **1.03×** | 1.83× | 28 |
| 2d | Float32 | **1.15×** | 2.12× | 18 |
| 3d | Float64 | **1.03×** | 2.84× | 10 |
| 3d | Float32 | **1.00×** | 3.84× | 10 |
| batched_dim1 | Float64 | **1.13×** | 1.41× | 15 |
| batched_dim1 | Float32 | **1.58×** | 1.60× | 12 |
| batched_dim2 | Float64 | **1.02×** | 1.18× | 12 |
| batched_dim2 | Float32 | **1.05×** | 1.30× | 12 |

**all: 1.17× vs int** (worst 0.36×, best 2.39×, 505 cases)
