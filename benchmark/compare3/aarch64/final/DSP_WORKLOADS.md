# DSP.jl workloads, FFTW vs FFTA integration/experiments (aarch64, 1 thread, DSP.jl ffta-eval branch)

```
== FFTW
welch_pgram(x,1024,512) n=2^18     1.981 ms
periodogram(x) n=2^16              0.826 ms
conv(a,b) 2^16*2^12                3.507 ms
conv(512x512, 16x16)               5.062 ms
conv(64^3, 8^3)                    10.857 ms
fftfilt(b257, x) n=2^18            2.589 ms
stft(x,512,256) n=2^17             1.052 ms
mt_pgram(x) n=2^14                 40.45 ms
hilbert(x) n=2^17                  3.735 ms
== FFTA integration/experiments
welch_pgram(x,1024,512) n=2^18     2.516 ms
periodogram(x) n=2^16              1.449 ms
conv(a,b) 2^16*2^12                4.014 ms
conv(512x512, 16x16)               7.774 ms
conv(64^3, 8^3)                    26.035 ms
fftfilt(b257, x) n=2^18            3.251 ms
stft(x,512,256) n=2^17             1.343 ms
mt_pgram(x) n=2^14                 40.652 ms
hilbert(x) n=2^17                  6.112 ms
```
