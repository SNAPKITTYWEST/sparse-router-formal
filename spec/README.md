# NraayTensor Specification Suite

## Overview

This directory contains the **normative specifications** for the NraayTensor hardware-oriented tensor governance system. These documents define the contract that both Rust and Pascal implementations must conform to.

All specifications are derived from Ahmad's tensor semantic framework with temporal safety extensions (BLRD model).

---

## Document Index

### 1. [TENSOR_CONTRACT.md](TENSOR_CONTRACT.md)
**Purpose**: Complete operation interface specification

**Contents**:
- TensorDescriptor C-compatible layout (76 bytes, 8-byte aligned)
- 8 normative operations: ALLOC, FREE, VIEW, TRANSPOSE, RESHAPE, MATERIALIZE, GET, SET
- Pre-conditions and post-conditions for each operation
- Ownership effects (Arc refcount management)
- Memory effects (buffer allocation, offset, stride computation)
- Failure conditions and error handling
- Pseudocode contract summary

**Key Concepts**:
```
OP-1: ALLOC — Create tensor, B_layout=1, B_own=1, refcount=1
OP-2: FREE  — Destructor, Arc::drop, buffer deallocation when refcount→0
OP-3: VIEW  — Zero-copy slicing, new offset/strides, B_own→0, refcount++
OP-4: TRANSPOSE — Logical permutation, metadata-only, recompute B_layout
OP-5: RESHAPE — Safe size-preserving shape change (contiguous/exclusive only)
OP-6: MATERIALIZE — COW-based contiguity/exclusivity restoration
OP-7: GET  — Element access (read), no state change
OP-8: SET  — Element access (write), COW if refcount>1
```

**Critical Invariants**:
1. Physical buffer size never shrinks during view operations
2. All address computations bounded by buffer length
3. B_layout ⟺ canonical stride pattern
4. B_own ⟺ refcount = 1
5. Fail-closed: precondition failures leave tensor unchanged

---

### 2. [GOVERNANCE_STATE_MACHINE.md](GOVERNANCE_STATE_MACHINE.md)
**Purpose**: State space, transitions, and invariants

**Contents**:
- 4-state space: (B_layout, B_own) ∈ {(0,0), (0,1), (1,0), (1,1)}
- State diagram with all legal transitions
- Pre/post-conditions for every transition
- Canonical form definition and canonical strides formula
- Generation counter management (u64 for staleness detection)
- Invariant conditions (must hold in all states)
- Reachability matrix showing valid transition paths

**State Space**:
```
State 00: SHARED_NON_CANONICAL    — Multiple views, non-contiguous
State 01: SHARED_CANONICAL        — Multiple views, contiguous (rare)
State 10: EXCLUSIVE_NON_CANONICAL — Single owner, non-contiguous
State 11: EXCLUSIVE_CANONICAL     — Single owner, contiguous (creation state)
```

**Canonical Strides**:
```
For shape (s₀, s₁, ..., s_{d-1}):
  σᵢ = ∏_{j>i} sⱼ

Examples:
  [2, 3]     → [3, 1]
  [2, 3, 4]  → [12, 4, 1]
```

**Generation Counter**:
```
Incremented on materialization or exclusive write (COW)
Detects stale views after parent materialization
Check: view.generation < parent.generation → StaleFork error
```

---

### 3. [MEMORY_MODEL.md](MEMORY_MODEL.md)
**Purpose**: Physical storage layout and address generation

**Contents**:
- Physical storage (𝒫): 1D contiguous f32 array
- Semantic view (𝒮): (Shape, Offset, Strides) tuple
- Core index mapping formula: `phys_addr = offset + Σ(coord[i] * strides[i])`
- TensorDescriptor C-compatible binary layout (FFI-ready)
- Memory domain classification (Register, Local, Shared, Numa, Host, Device, Mmio)
- Stride/offset computation without framework overhead (O(d) where d=rank)
- Allocation patterns: dense, strided, sliced, transposed

**Index Mapping Formula**:
```
For logical coordinate X = (x₀, x₁, ..., x_{d-1}):
  physical_address(X) = offset + Σᵢ (xᵢ * strideᵢ)

Preconditions:
  - shape.len() = strides.len() = coord.len()
  - ∀ i: 0 ≤ xᵢ < sᵢ
  - physical_address < |𝒫|
```

**TensorDescriptor Layout** (76 bytes):
```
Offset   Size   Field                Type
0        8      data_ptr             *const f32
8        8      data_len             u64
16       8      shape_ptr            *const u64
24       8      shape_len            u64
32       8      strides_ptr          *const u64
40       8      strides_len          u64
48       8      offset               u64
56       1      is_contiguous        u8 (B_layout)
57       1      owns_storage         u8 (B_own)
58       2      _padding             u16
60       8      generation           u64
68       8      refcount (snapshot)  u64
```

---

### 4. [TEMPORAL_SAFETY.md](TEMPORAL_SAFETY.md)
**Purpose**: Lifetime proofs and use-after-free prevention

**Contents**:
- Lifetime algebra: ℒ_v (view lifetime), ℒ_ℬ (buffer lifetime), ℒ_𝓞 (Arc lifetime)
- Core temporal relation: `ℒ_v ⊆ ℒ_𝓞 ⊆ ℒ_ℬ`
- 5 executable lemmas with test implementations:
  - **Lemma 1**: Buffer lifetime safety (Arc::strong_count > 0 → buffer valid)
  - **Lemma 2**: Non-interference of shared views (COW isolation)
  - **Lemma 3**: Exclusive post-materialization (refcount=1 after materialize)
  - **Lemma 4**: Governance consistency (flags ⟺ state)
  - **Lemma 5**: Weak observer invariant (downgrade doesn't extend lifetime)
- Generation validation rules (staleness detection)
- Use-after-release rejection mechanisms
- COW isolation theorem with proof and executable test

**Core Theorem**: Buffer Lifetime Safety
```
∀ view V holding Arc<Vec<f32>>:
  Arc::strong_count(V.data) ≥ 1 while V is live
  ⟹ buffer remains allocated and accessible
  ⟹ lifetime(V) ⊆ lifetime(buffer(V)) ∎
```

**COW Isolation Theorem**:
```
If V_A and V_B forked from P, and V_A materializes:
  V_A.data = new Arc (exclusive)
  V_B.data = original Arc (still shared with P)
  ⟹ No interference between V_A and V_B
```

---

### 5. [PROOF_OBLIGATIONS.md](PROOF_OBLIGATIONS.md)
**Purpose**: Formal lemmas (PO-1 to PO-8) with test oracles

**Contents**:
- 8 proof obligations with formal statements and executable tests
- Test strategy for each obligation
- Cross-implementation validation approach
- Oracle behavior specifications (what Rust and Pascal must produce)

**Proof Obligations**:
```
PO-1: Contiguity Preservation
  Physical buffer size never shrinks, all addresses valid

PO-2: Mapping Uniqueness
  Index mapping is injective (no coordinate collisions)

PO-3: Governance Correctness
  B_layout ⟺ canonical strides, B_own ⟺ refcount=1

PO-4: Slice Soundness
  Slice coordinates map correctly to parent's physical addresses

PO-5: COW Isolation
  Materialization creates isolation, siblings unaffected

PO-6: Temporal Safety
  lifetime(view) ⊆ lifetime(buffer) via Arc::strong_count

PO-7: Materialization Correctness
  Values preserved, contiguity/exclusivity restored

PO-8: Generation Counter Validation
  Staleness detection after parent materialization
```

---

## Reading Guide

### For Implementers (Rust, Pascal)

1. **Start here**: [TENSOR_CONTRACT.md](TENSOR_CONTRACT.md)
   - Understand the 8 operations and their contracts
   - Implement each operation to match pre/post-conditions

2. **Then read**: [GOVERNANCE_STATE_MACHINE.md](GOVERNANCE_STATE_MACHINE.md)
   - Implement state transitions correctly
   - Maintain B_layout and B_own flags accurately
   - Manage generation counters

3. **Reference**: [MEMORY_MODEL.md](MEMORY_MODEL.md)
   - Implement address generation formula
   - Layout tensor descriptors for FFI
   - Understand memory domains for optimization

4. **Verify**: [TEMPORAL_SAFETY.md](TEMPORAL_SAFETY.md) + [PROOF_OBLIGATIONS.md](PROOF_OBLIGATIONS.md)
   - Run all 8 proof obligation tests
   - Verify against oracle behavior
   - Cross-check with other implementation

### For Architecture Review

Read in this order:
1. TENSOR_CONTRACT.md (high-level operations)
2. GOVERNANCE_STATE_MACHINE.md (state space safety)
3. TEMPORAL_SAFETY.md (lifetime guarantees)
4. MEMORY_MODEL.md (physical layout correctness)
5. PROOF_OBLIGATIONS.md (formal verification)

---

## Critical Invariants (Must Hold Everywhere)

1. **B_layout Correctness**: `B_layout = 1 ⟺ strides == canonical_strides(shape)`
2. **B_own Correctness**: `B_own = 1 ⟺ Arc::strong_count(data) = 1`
3. **Buffer Validity**: `offset < data.len() ∧ ∀ coord: addr(coord) < data.len()`
4. **Ownership Protection**: Shared tensors (B_own=0) cannot be exclusively modified without COW
5. **Materialization Correctness**: After MATERIALIZE, B_layout=1 ∧ B_own=1 ∧ offset=0
6. **Temporal Safety**: Buffer remains allocated while any view holds Arc reference
7. **Generation Monotonicity**: generation only increases (never resets)

---

## Fail-Closed Design

All operations adopt fail-closed semantics:

```
If precondition(op) fails:
  tensor state unchanged
  error returned
Else:
  operation executes
  postconditions guaranteed
  state update atomic
```

This ensures no partial updates or corrupted state on error.

---

## Implementation Checklist

### Rust Implementation
- [ ] Implement 8 operations matching TENSOR_CONTRACT.md
- [ ] Manage Arc lifecycle and refcount correctly
- [ ] Compute canonical strides per MEMORY_MODEL.md formula
- [ ] Implement COW in SET and MATERIALIZE
- [ ] Generate indices O(1) for rank 1, O(d) for rank d
- [ ] Pass all PO-1 through PO-8 tests from PROOF_OBLIGATIONS.md
- [ ] Verify temporal safety lemmas in TEMPORAL_SAFETY.md

### Pascal Implementation
- [ ] Implement reference counting (refcount management)
- [ ] Match TensorDescriptor C-compatible layout (MEMORY_MODEL.md)
- [ ] Call Rust via FFI or reimplement 8 operations identically
- [ ] Pass same oracle tests as Rust
- [ ] Verify cross-implementation consistency

---

## References

### External Standards
- Ahmad's Semantic Tensor Framework (original theory)
- BLRD Temporal Extension (lifetime algebra)
- C99 ABI (TensorDescriptor FFI layout)

### Related Documents
- NraayTensor README: ../nraay-tensor/README.md
- Implementation Notes: ../nraay-tensor/src/
- Test Suite: ../nraay-tensor/tests/

---

## Document Maintenance

**Version**: 1.0 (2025-09-22)  
**Authority**: Architecture Designer (Hardware-Oriented Tensor Governance)  
**License**: GPL-3.0 / Apache-2.0 / MIT (triple license per commercial repo policy)

---

## Quick Reference: State Machine Diagram

```
           Creation (ALLOC)
                 |
                 v
        ┌────────────────┐
        │ 11: Exclusive  │
        │   Canonical    │
        │ B_layout=1     │
        │ B_own=1        │
        └────┬───────┬───┘
             |       |
          VIEW()  TRANSPOSE()
             |       |
    ┌────────v───┐ ┌─v──────────┐
    │ 00: Shared │ │ 10: Excl.  │
    │ Non-Canon. │ │ Non-Canon. │
    │ B_own=0    │ │ B_layout=0 │
    └────────┬───┘ └─┬──────────┘
             |       |
             └─────┬─┘
                   |
            MATERIALIZE()
                   |
                   v
        ┌────────────────┐
        │ 11: Exclusive  │
        │   Canonical    │
        │ (always)       │
        └────────────────┘
```

---

## Summary

This specification suite provides:
- ✓ Complete operation interface (8 ops with pre/post-conditions)
- ✓ State machine correctness (4 states, valid transitions)
- ✓ Memory layout and FFI compatibility
- ✓ Temporal safety guarantees (no use-after-free)
- ✓ Formal verification (8 proof obligations with tests)
- ✓ Cross-implementation consistency (oracle tests)

**Implementations must pass all proof obligations and oracle tests to be conformant.**
