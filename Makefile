CC ?= cc
BUILD_DIR ?= build

BREW_PREFIX := $(shell command -v brew >/dev/null 2>&1 && brew --prefix 2>/dev/null)
LLVM_PREFIX := $(firstword $(wildcard $(BREW_PREFIX)/opt/llvm $(BREW_PREFIX)/opt/llvm@22))

ifneq ($(LLVM_PREFIX),)
  OMP_CC ?= $(LLVM_PREFIX)/bin/clang
  OMP_LDFLAGS ?= -L$(LLVM_PREFIX)/lib -Wl,-rpath,$(LLVM_PREFIX)/lib
else
  OMP_CC ?= $(CC)
  OMP_LDFLAGS ?=
endif

COMMON_FLAGS ?= -std=c11 -Wall -Wextra -march=native
FAST_FLAGS ?= -O3 -ffast-math
OMP_FLAGS ?= -fopenmp

.PHONY: all run runfast runomp runneon runsimd runq debug win64 rungnu runompgnu test testc testcc clean benchmark

all: run

# Upstream-compatible, standards-conforming single-thread build.
run: run.c
	$(CC) $(COMMON_FLAGS) -O3 run.c -lm -o run

# Same source with relaxed floating-point semantics, enabling reassociated SIMD.
runfast: run.c
	$(CC) $(COMMON_FLAGS) $(FAST_FLAGS) run.c -lm -o run-fast

# Auto-vectorized OpenMP build. OMP_SCHEDULE selects static/dynamic/guided.
runomp: run.c
	$(OMP_CC) $(COMMON_FLAGS) $(FAST_FLAGS) $(OMP_FLAGS) \
		-DFASTOLLAMA_RUNTIME_SCHEDULE $(OMP_LDFLAGS) run.c -lm -o run-omp

# Explicit ARM NEON experiment, compiled with the same fast-math/OpenMP flags.
runneon: run.c
	$(OMP_CC) $(COMMON_FLAGS) $(FAST_FLAGS) $(OMP_FLAGS) \
		-DFASTOLLAMA_RUNTIME_SCHEDULE -DFASTOLLAMA_NEON \
		$(OMP_LDFLAGS) run.c -lm -o run-neon

# Compatibility alias for the name used during the original experiment.
runsimd: runneon

runq: runq.c
	$(CC) $(COMMON_FLAGS) -O3 runq.c -lm -o runq

debug: run.c
	$(CC) $(COMMON_FLAGS) -O0 -g run.c -lm -o run

# Inherited cross-platform targets from llama2.c.
win64:
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o run.exe -I. run.c win.c
	x86_64-w64-mingw32-gcc -Ofast -D_WIN32 -o runq.exe -I. runq.c win.c

rungnu:
	$(CC) -Ofast -std=gnu11 -o run run.c -lm
	$(CC) -Ofast -std=gnu11 -o runq runq.c -lm

runompgnu:
	$(CC) -Ofast -fopenmp -std=gnu11 run.c -lm -o run
	$(CC) -Ofast -fopenmp -std=gnu11 runq.c -lm -o runq

test: run
	pytest

testc:
	$(CC) -DVERBOSITY=0 -O3 -o testc test.c -lm
	./testc

testcc: testc

benchmark:
	@test -n "$(MODEL)" || (echo "usage: make benchmark MODEL=/path/to/model.bin"; exit 2)
	python3 benchmarks/run_benchmarks.py --model "$(MODEL)"

clean:
	rm -rf $(BUILD_DIR) run run-fast run-omp run-neon run-simd runq run.exe runq.exe testc
