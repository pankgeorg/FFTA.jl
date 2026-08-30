### leaves vs int (speedup, >1 = faster) and leaves/fftw (ratio, lower = better)

| group | type | speedup vs int | leaves/fftw | cases |
|:--|:--|--:|--:|--:|
| pow2 | Float64 | **1.25×** | 1.49× | 40 |
| pow2 | Float32 | **1.68×** | 1.26× | 40 |
| smooth | Float64 | **0.98×** | 4.29× | 48 |
| smooth | Float32 | **0.98×** | 6.74× | 48 |
| prime | Float64 | **1.19×** | 1.12× | 48 |
| prime | Float32 | **1.64×** | 1.01× | 48 |
| awkward | Float64 | **1.21×** | 1.42× | 58 |
| awkward | Float32 | **1.79×** | 1.47× | 58 |
| 2d | Float64 | **1.04×** | 1.81× | 28 |
| 2d | Float32 | **1.15×** | 2.14× | 18 |
| 3d | Float64 | **1.01×** | 2.90× | 10 |
| 3d | Float32 | **0.98×** | 3.92× | 10 |
| batched_dim1 | Float64 | **1.05×** | 1.52× | 15 |
| batched_dim1 | Float32 | **1.39×** | 1.81× | 12 |
| batched_dim2 | Float64 | **0.96×** | 1.26× | 12 |
| batched_dim2 | Float32 | **0.98×** | 1.38× | 12 |

**all: 1.25× vs int** (worst 0.53×, best 4.67×, 505 cases)
