# Sparse Router — Formally Verified Locality-Aware Work-Stealing Scheduler

**License**: GPL-3.0-or-later OR Apache-2.0 (dual-licensed)

## Architecture

```
sparse-router-formal/
├── rust/tensor-hw/       # Production Rust: tensor core + RAW_ROUTE + Kani proofs
│   ├── src/              # 10 modules: buffer, ownership, memory, view, governance...
│   └── ffi/              # sparse_router.rs, kani proofs, C ABI, FFI bridge
├── pascal/               # Reference implementation (independent conformance oracle)
├── formal/               # Lean 4 formal proofs (convergence, invariants, scoring)
├── c-reference/          # Standalone C reference (header-only, benchmarkable)
├── tests/conformance/    # 52 cross-language conformance tests (7 categories)
├── spec/                 # Normative specifications (contract, state machine, memory model)
├── src/pascal/           # Pascal transformer sources
└── scripts/              # Clone-gate verification
```

## Novel Method: RAW_ROUTE

Three-phase routing with O(K) sparse-edge traversal (K ≤ 16), not O(N) worker scan:

```
score(w, t) = locality(w,t) + memory_affinity(w,t) + queue_pressure(w)
            + stealability(w) - migration_cost(w,t)
```

| Phase | Path | Complexity |
|-------|------|-----------|
| 1 | Region-owner fast path | O(1) |
| 2 | Sparse edge traversal | O(K) |
| 3 | Least-loaded fallback | O(N) |

## Formal Verification

### Lean 4 Proofs (`formal/`)
- **Convergence**: potential function Φ = |unowned regions| strictly decreases → stable in ≤ MAX_REGIONS steps
- **Uniqueness**: every routed task lands on exactly one worker deque
- **Bounded traversal**: sparse edge fan-out ≤ MAX_SPARSE_EDGES
- **Scoring properties**: locality bonus, empty-queue bonus, zero migration cost for local tasks
- **Deque correctness**: push/pop identity, count invariants

### Kani Proofs (`rust/tensor-hw/ffi/sparse_router_kani.rs`)
- All tasks routed (progress guaranteed)
- Region owner consistency
- Locality over balance
- Migration cost avoidance
- Fallback coverage
- Route decision stability
- Score bounds (all weights finite)
- No deadlock (acyclic region ownership)

### Cross-Language Conformance (`tests/conformance/`)
52 tests across 7 categories — Rust and Pascal must agree:
- Buffer lifetime, ownership transitions, views, materialization
- Generation validation, governance, routing

## Build

```bash
make test              # C reference tests (7 tests)
make bench             # C benchmarks (~100M routes/sec)
make test-rust         # Rust test suite (49 tests)
make verify            # Lean 4 proofs (requires lake)
make clone-gate-check  # Verify all markers present, no MIT
```

## Clone Gate

All source files contain `CLONE_GATE` markers. See `CLONE_GATE.md` for terms.
No MIT licensing permitted. See `LICENSE-GPL` and `LICENSE-APACHE`.

## Deliverables Summary

| Component | LOC | Status |
|-----------|-----|--------|
| Specifications | ~3,600 | 7 normative documents |
| Rust Core | ~2,670 | 10 modules + 49 tests |
| Pascal Reference | ~2,450 | 10 modules + 10 tests |
| Lean 4 Proofs | ~350 | 6 modules, convergence theorem |
| C Reference | ~450 | Header-only + benchmarks |
| Conformance Tests | ~2,800 | 52 tests + oracle |
| Kani Proofs | ~400 | 8 formal proofs |
| **Total** | **~12,700** | **Production ready** |
