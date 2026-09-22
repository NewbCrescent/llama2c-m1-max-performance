# Benchmark summary

- CPU: Apple M1 Max
- Compiler: Homebrew clang version 23.1.0
- Model SHA-256: `515267168726a1ed1317a64a408492e6af3b67c1f71c5bd98c01d9d721803a24` (438.4 MB)
- Method: 1 warmup(s), 5 measured trials, 64 generated tokens
- Best observed configuration: neon / 8 threads / static at 136.96 tok/s (14.49x baseline)

| Variant | Threads | Schedule | Median tok/s | Mean ± SD | Speedup | Same output |
|---|---:|---|---:|---:|---:|:---:|
| baseline | 1 | - | 9.45 | 9.41 ± 0.09 | 1.00x | yes |
| fast | 1 | - | 95.74 | 94.91 ± 1.98 | 10.13x | yes |
| omp | 1 | static | 92.78 | 92.98 ± 0.53 | 9.82x | yes |
| omp | 2 | static | 118.87 | 118.24 ± 2.35 | 12.58x | yes |
| omp | 4 | static | 121.15 | 121.44 ± 0.92 | 12.82x | yes |
| omp | 8 | static | 135.19 | 135.22 ± 2.04 | 14.30x | yes |
| omp | 10 | static | 123.29 | 124.12 ± 2.91 | 13.05x | yes |
| omp | 1 | dynamic | 92.78 | 92.84 ± 0.63 | 9.82x | yes |
| omp | 2 | dynamic | 48.31 | 47.93 ± 0.77 | 5.11x | yes |
| omp | 4 | dynamic | 64.22 | 64.31 ± 0.29 | 6.80x | yes |
| omp | 8 | dynamic | 70.00 | 69.67 ± 1.01 | 7.41x | yes |
| omp | 10 | dynamic | 62.38 | 62.29 ± 1.04 | 6.60x | yes |
| omp | 1 | guided | 93.61 | 93.45 ± 0.82 | 9.90x | yes |
| omp | 2 | guided | 123.05 | 122.72 ± 1.31 | 13.02x | yes |
| omp | 4 | guided | 119.32 | 117.36 ± 4.91 | 12.63x | yes |
| omp | 8 | guided | 122.81 | 122.76 ± 2.78 | 12.99x | yes |
| omp | 10 | guided | 110.53 | 106.33 ± 9.87 | 11.69x | yes |
| neon | 1 | static | 91.57 | 90.86 ± 1.82 | 9.69x | yes |
| neon | 2 | static | 117.10 | 114.87 ± 6.28 | 12.39x | yes |
| neon | 4 | static | 121.62 | 120.69 ± 2.57 | 12.87x | yes |
| neon | 8 | static | 136.96 | 134.82 ± 4.86 | 14.49x | yes |
| neon | 10 | static | 127.27 | 126.36 ± 2.97 | 13.47x | yes |

`baseline` is `-O3`; `fast` adds `-ffast-math`; `omp` adds OpenMP; `neon` replaces the hot dot-product loop with explicit ARM NEON intrinsics.
