# DSP.jl workloads, FFTW vs FFTA integration/experiments @ 57ce6e3 (aarch64; DSP.jl ffta-eval branch; FFTW with set_num_threads(nthreads), FFTA plans default num_threads)

## 1 thread(s)
```
== FFTW
welch_pgram(x,1024,512) n=2^18     1.968 ms
periodogram(x) n=2^16              0.835 ms
conv(a,b) 2^16*2^12                3.555 ms
conv(512x512, 16x16)               4.359 ms
conv(64^3, 8^3)                    11.402 ms
fftfilt(b257, x) n=2^18            2.575 ms
stft(x,512,256) n=2^17             1.055 ms
mt_pgram(x) n=2^14                 40.408 ms
hilbert(x) n=2^17                  3.728 ms
== FFTA
welch_pgram(x,1024,512) n=2^18     2.527 ms
periodogram(x) n=2^16              1.169 ms
conv(a,b) 2^16*2^12                3.953 ms
conv(512x512, 16x16)               7.852 ms
conv(64^3, 8^3)                    26.442 ms
fftfilt(b257, x) n=2^18            3.262 ms
stft(x,512,256) n=2^17             1.352 ms
mt_pgram(x) n=2^14                 40.492 ms
hilbert(x) n=2^17                  6.18 ms
```

## 4 thread(s)
```
== FFTW
welch_pgram(x,1024,512) n=2^18     1.985 ms
periodogram(x) n=2^16              0.719 ms
conv(a,b) 2^16*2^12                3.368 ms
conv(512x512, 16x16)               5.327 ms
conv(64^3, 8^3)                    12.782 ms
fftfilt(b257, x) n=2^18            9.091 ms
stft(x,512,256) n=2^17             1.016 ms
mt_pgram(x) n=2^14                 41.122 ms
hilbert(x) n=2^17                  2.765 ms
== FFTA
welch_pgram(x,1024,512) n=2^18     2.566 ms
periodogram(x) n=2^16              1.186 ms
conv(a,b) 2^16*2^12                4.168 ms
conv(512x512, 16x16)               7.69 ms
conv(64^3, 8^3)                    25.326 ms
fftfilt(b257, x) n=2^18            3.328 ms
stft(x,512,256) n=2^17             1.374 ms
mt_pgram(x) n=2^14                 40.563 ms
hilbert(x) n=2^17                  7.82 ms
```

## 16 thread(s)
```
== FFTW
welch_pgram(x,1024,512) n=2^18     1.983 ms
periodogram(x) n=2^16              0.67 ms
conv(a,b) 2^16*2^12                2.702 ms
conv(512x512, 16x16)               5.975 ms
conv(64^3, 8^3)                    12.651 ms
fftfilt(b257, x) n=2^18            5.924 ms
stft(x,512,256) n=2^17             1.071 ms
mt_pgram(x) n=2^14                 41.038 ms
hilbert(x) n=2^17                  2.375 ms
== FFTA
welch_pgram(x,1024,512) n=2^18     2.59 ms
periodogram(x) n=2^16              1.582 ms
conv(a,b) 2^16*2^12                6.308 ms
conv(512x512, 16x16)               7.986 ms
conv(64^3, 8^3)                    25.79 ms
fftfilt(b257, x) n=2^18            3.331 ms
stft(x,512,256) n=2^17             1.388 ms
mt_pgram(x) n=2^14                 40.738 ms
hilbert(x) n=2^17                  9.474 ms
```
