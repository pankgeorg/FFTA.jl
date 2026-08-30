impls=['fftw', 'int', 'poly'] threads=[1, 4, 6, 8, 12]

## poly vs int (>1 = poly faster), geomean over matched cases
| threads | all | batched d=1 | batched d=2 | 2D/3D | worst | best | cases |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1 | **1.01×** | 0.98× | 1.02× | 1.02× | 0.77× | 1.50× | 117 |
| 4 | **1.07×** | 1.08× | 1.04× | 1.07× | 0.82× | 1.53× | 117 |
| 6 | **1.12×** | 1.18× | 1.04× | 1.14× | 0.62× | 3.22× | 117 |
| 8 | **1.22×** | 1.29× | 1.15× | 1.22× | 0.54× | 2.90× | 117 |
| 12 | **1.29×** | 1.26× | 1.21× | 1.33× | 0.57× | 6.09× | 117 |

## thread scaling (t1 time / tN time), geomean; large cases only (>=THREAD_THRESHOLD=32768 elems)
| threads | `fftw` | `int` | `poly` |
|--:|--:|--:|--:|
| 4 | 2.95× | 3.13× | 3.47× |
| 6 | 3.51× | 3.89× | 4.69× |
| 8 | 3.71× | 4.51× | 6.15× |
| 12 | 4.17× | 4.59× | 6.89× |

## straggler check: poly/int at each thread count, large cases only
  t=1   poly vs int 1.01×  (worst 0.81×, best 1.50×, n=71)
  t=4   poly vs int 1.12×  (worst 0.85×, best 1.53×, n=71)
  t=6   poly vs int 1.22×  (worst 0.62×, best 3.22×, n=71)
  t=8   poly vs int 1.38×  (worst 0.54×, best 2.90×, n=71)
  t=12  poly vs int 1.52×  (worst 0.57×, best 6.09×, n=71)

## checksum agreement (poly vs int)
  all match
