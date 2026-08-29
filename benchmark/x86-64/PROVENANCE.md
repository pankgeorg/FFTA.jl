# Commit provenance for the x86-64 runs

Measured on: Intel Core Ultra 7 165H (alderlake), Julia 1.12.6, FFTW 3.3.11 (provider `fftw`), -t 8.
All runs use the *same* `suite.jl`/`report.jl`/`compare.jl` copied from
pankgeorg/FFTA.jl `bench/comprehensive-suite` @ 73b190c, so before/after pairs are
methodologically identical. (The bench branch later moved to d0064b1, which adds a
`"complete": true` flag and line-flushing but changes nothing that is measured;
deliberately not adopted mid-sequence.)

| run | branch | commit measured | note |
|:--|:--|:--|:--|
| baseline | bench/comprehensive-suite | 73b190c | FFTA `src/` byte-identical to JuliaMath/FFTA.jl main @ 7aeb327 (FFTA 0.3.1); verified with `git diff origin/main...HEAD -- src/ Project.toml` (empty) |
| integration_all | integration/all | b06c41c | branch head later moved to 45696ac; `git diff b06c41c 45696ac -- src/ Project.toml` is empty (test-only changes), so these numbers are valid for 45696ac |
| A_plan_twiddles | feat/plan-twiddles | 8411861 | |
| C_pow2_codelets | feat/pow2-codelets | 5b02c9d | `--only 1d --kinds fft` |
| B_real_plan_mul | feat/real-plan-mul | c9d0a99 | `--kinds rfft` |
| D_nd_buffers | feat/nd-buffers-threads | 17ed942 | `--only nd,batched,threads` |
| E_bluestein | feat/bluestein-padding | 919628e | `--only 1d --kinds fft` |
