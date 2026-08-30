### odd vs int (speedup, >1 = faster) and odd/fftw (ratio, lower = better)

| group | type | speedup vs int | odd/fftw | cases |
|:--|:--|--:|--:|--:|
| pow2 | Float64 | **1.00×** | 1.85× | 40 |
| pow2 | Float32 | **1.01×** | 2.10× | 40 |
| smooth | Float64 | **1.46×** | 2.88× | 48 |
| smooth | Float32 | **1.79×** | 3.70× | 48 |
| prime | Float64 | **1.02×** | 1.31× | 48 |
| prime | Float32 | **1.05×** | 1.56× | 48 |
| awkward | Float64 | **0.91×** | 1.89× | 58 |
| awkward | Float32 | **0.94×** | 2.79× | 58 |
| 2d | Float64 | **1.05×** | 1.78× | 28 |
| 2d | Float32 | **1.00×** | 2.44× | 18 |
| 3d | Float64 | **0.99×** | 2.96× | 10 |
| 3d | Float32 | **0.96×** | 4.00× | 10 |
| batched_dim1 | Float64 | **0.97×** | 1.65× | 15 |
| batched_dim1 | Float32 | **0.99×** | 2.56× | 12 |
| batched_dim2 | Float64 | **0.92×** | 1.32× | 12 |
| batched_dim2 | Float32 | **0.96×** | 1.41× | 12 |

**all: 1.08× vs int** (worst 0.50×, best 7.21×, 505 cases)
