# Original dot-product experiment

The first optimization pass used a standalone integer dot-product workload to
learn ARM NEON intrinsics and compare OpenMP schedules. On its largest input
(100 million elements repeated 100 times), the saved measurements were:

| Configuration | Serial | Parallel | Speedup |
|---|---:|---:|---:|
| 10 threads, static | 31.888 s | 4.396 s | 7.25x |
| 8 threads, static | 31.994 s | 4.432 s | 7.22x |
| 10 threads, dynamic (chunk 8192) | 32.054 s | 3.922 s | 8.17x |
| 10 threads, guided | 31.983 s | 3.754 s | 8.52x |

These results are retained as project history, not presented as Llama inference
performance. The workload performed 16 rounds of additional integer arithmetic
per element, used a single run per configuration, and hard-coded its thread
count. It was useful for forming hypotheses about scheduling, but it did not
model the full inference program closely enough to support an end-to-end speedup
claim.

The exact source and raw saved output remain available in Git history at commit
`bce2780`. The current benchmark harness replaces it with repeated end-to-end
inference trials, deterministic output checks, compiler metadata, and raw data.
