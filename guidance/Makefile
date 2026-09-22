TARGET ?= fastollama
BUILD_DIR ?= build

SRCS := hello.c arr.c
OBJS := $(SRCS:%.c=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

BREW_PREFIX := $(shell command -v brew >/dev/null 2>&1 && brew --prefix 2>/dev/null)
LLVM_PREFIX := $(firstword $(wildcard $(BREW_PREFIX)/opt/llvm $(BREW_PREFIX)/opt/llvm@22))

CC ?= cc
CPPFLAGS ?=
CFLAGS ?= -std=c11 -O3 -Wall -Wextra -MMD -MP
LDFLAGS ?=
LDLIBS ?=
OPENMP ?= 1

ifeq ($(OPENMP),1)
  ifneq ($(LLVM_PREFIX),)
    CC := $(LLVM_PREFIX)/bin/clang
    CFLAGS += -fopenmp
    LDFLAGS += -L$(LLVM_PREFIX)/lib -Wl,-rpath,$(LLVM_PREFIX)/lib
    LDLIBS += -lomp
  else
    CFLAGS += -fopenmp
    LDFLAGS += -fopenmp
  endif
endif

.PHONY: all run check clean help

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@

$(BUILD_DIR)/%.o: %.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

$(BUILD_DIR):
	mkdir -p $@

run: $(TARGET)
	./$(TARGET)

check: run

clean:
	rm -rf $(BUILD_DIR) $(TARGET) a.out

help:
	@echo "Targets:"
	@echo "  make              Build $(TARGET)"
	@echo "  make run          Build and run $(TARGET)"
	@echo "  make check        Same as run"
	@echo "  make clean        Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  make OPENMP=1     Build with OpenMP support"
	@echo "  make TARGET=name  Change output binary name"

-include $(DEPS)
