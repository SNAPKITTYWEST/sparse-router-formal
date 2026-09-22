# SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
# Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
# CLONE_GATE: sparse-router-formal::makefile

CC      := gcc
CFLAGS  := -O2 -Wall -Wextra -Wpedantic -std=c11

.PHONY: all test bench verify clone-gate-check clean

all: build/router build/test_router

build/router: c-reference/main.c c-reference/router.h c-reference/types.h c-reference/scoring.h c-reference/deque.h
	@mkdir -p build
	$(CC) $(CFLAGS) -I c-reference -o $@ c-reference/main.c

build/test_router: c-reference/tests/test_router.c c-reference/router.h
	@mkdir -p build
	$(CC) $(CFLAGS) -I c-reference -o $@ c-reference/tests/test_router.c

test: build/test_router
	./build/test_router

bench: build/router
	./build/router

test-rust:
	cd rust/tensor-hw && cargo test

verify:
	cd formal && lake build

clone-gate-check:
	@bash scripts/clone_gate_verify.sh

clean:
	rm -rf build/
