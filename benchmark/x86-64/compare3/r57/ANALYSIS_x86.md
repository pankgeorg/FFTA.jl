### r57 vs int (speedup, >1 = faster) and r57/fftw (ratio, lower = better)

| group | type | speedup vs int | r57/fftw | cases |
|:--|:--|--:|--:|--:|
| pow2 | Float64 | **1.01×** | 1.84× | 40 |
| pow2 | Float32 | **1.00×** | 2.12× | 40 |
| smooth | Float64 | **1.50×** | 2.81× | 48 |
| smooth | Float32 | **1.77×** | 3.74× | 48 |
| prime | Float64 | **1.08×** | 1.24× | 48 |
| prime | Float32 | **1.10×** | 1.50× | 48 |
| awkward | Float64 | **0.94×** | 1.83× | 58 |
| awkward | Float32 | **0.97×** | 2.70× | 58 |
| 2d | Float64 | **1.03×** | 1.82× | 28 |
| 2d | Float32 | **0.97×** | 2.53× | 18 |
| 3d | Float64 | **1.00×** | 2.91× | 10 |
| 3d | Float32 | **0.98×** | 3.92× | 10 |
| batched_dim1 | Float64 | **1.02×** | 1.57× | 15 |
| batched_dim1 | Float32 | **1.02×** | 2.49× | 12 |
| batched_dim2 | Float64 | **1.00×** | 1.21× | 12 |
| batched_dim2 | Float32 | **0.98×** | 1.39× | 12 |

**all: 1.11× vs int** (worst 0.62×, best 7.68×, 505 cases)
