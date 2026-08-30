# DSP.jl workloads, FFTW vs FFTA (integration/experiments + lazy workers), aarch64; DSP.jl ffta-eval branch; FFTW with set_num_threads(nthreads), FFTA plans default num_threads

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
welch_pgram(x,1024,512) n=2^18     2.536 ms
periodogram(x) n=2^16              1.5 ms
conv(a,b) 2^16*2^12                4.178 ms
conv(512x512, 16x16)               7.894 ms
conv(64^3, 8^3)                    26.733 ms
fftfilt(b257, x) n=2^18            3.277 ms
stft(x,512,256) n=2^17             1.365 ms
mt_pgram(x) n=2^14                 40.533 ms
hilbert(x) n=2^17                  6.031 ms
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
welch_pgram(x,1024,512) n=2^18     2.535 ms
periodogram(x) n=2^16              1.127 ms
conv(a,b) 2^16*2^12                4.09 ms
conv(512x512, 16x16)               7.913 ms
conv(64^3, 8^3)                    25.68 ms
fftfilt(b257, x) n=2^18            3.299 ms
stft(x,512,256) n=2^17             1.326 ms
mt_pgram(x) n=2^14                 40.486 ms
hilbert(x) n=2^17                  5.973 ms
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
welch_pgram(x,1024,512) n=2^18     2.538 ms
periodogram(x) n=2^16              1.046 ms
conv(a,b) 2^16*2^12                4.142 ms
conv(512x512, 16x16)               7.831 ms
conv(64^3, 8^3)                    25.714 ms
fftfilt(b257, x) n=2^18            3.263 ms
stft(x,512,256) n=2^17             1.357 ms
mt_pgram(x) n=2^14                 40.548 ms
hilbert(x) n=2^17                  6.102 ms
```
