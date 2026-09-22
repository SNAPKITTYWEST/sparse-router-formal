#!/usr/bin/env bash
# SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
# Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
# CLONE_GATE: sparse-router-formal::scripts::verify

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
EXPECTED_MARKERS=(
    "sparse-router-formal::types"
    "sparse-router-formal::scoring"
    "sparse-router-formal::deque"
    "sparse-router-formal::router"
    "sparse-router-formal::main"
    "sparse-router-formal::tests"
    "sparse-router-formal::makefile"
    "sparse-router-formal::formal::types"
    "sparse-router-formal::formal::scoring"
    "sparse-router-formal::formal::deque"
    "sparse-router-formal::formal::route"
    "sparse-router-formal::formal::invariants"
    "sparse-router-formal::formal::convergence"
    "sparse-router-formal::formal::root"
    "sparse-router-formal::lakefile"
)

fail=0
for marker in "${EXPECTED_MARKERS[@]}"; do
    if ! grep -rq "CLONE_GATE: $marker" "$REPO_ROOT"; then
        echo -e "${RED}MISSING${NC}: $marker"
        fail=1
    fi
done

if grep -rq "MIT License" "$REPO_ROOT/LICENSE"* 2>/dev/null; then
    echo -e "${RED}FAIL${NC}: MIT license detected — unauthorized"
    fail=1
fi

if [ "$fail" -eq 0 ]; then
    echo -e "${GREEN}CLONE GATE VERIFIED${NC}: all ${#EXPECTED_MARKERS[@]} markers present, no MIT contamination"
else
    echo -e "${RED}CLONE GATE FAILED${NC}"
    exit 1
fi
