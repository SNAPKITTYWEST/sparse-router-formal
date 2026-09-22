# NraayTensor FFI Contract

## Overview

The FFI layer provides a **C-compatible ABI** that bridges:
- **Rust `tensor.rs`** (production hardware layer)
- **Free Pascal** (reference implementation & oracle)
- **Formal verification** (Kani proofs for correctness)

All operations are defined through a canonical binary protocol (`tensor_abi.h`) that both implementations must conform to.

---

## Files

### C ABI Definition
- **`tensor_abi.h`** — Canonical C-compatible tensor descriptor and operation signatures
  - `TensorDescriptorC`: packed struct with buffer identity, rank, governance, ownership, generation
  - Operations: `tensor_allocate`, `tensor_view`, `tensor_materialize`, `tensor_load`, `tensor_store`, etc.

### Rust Implementation
- **`tensor_ffi.rs`** — Rust FFI layer that wraps the real `tensor.rs` implementation
  - Global registry (`TENSOR_REGISTRY`) maps buffer IDs to actual tensor data
  - Arc-based ownership semantics are preserved through FFI boundary
  - Every operation maps directly to tensor logic (no translation layer)

### Pascal Bindings
- **`tensor_abi.pas`** — Free Pascal FFI wrapper (calls into Rust via C ABI)
  - Pascal-idiomatic error handling with exception raising
  - Thin wrappers around C FFI calls
  - Same semantic contract as Rust

### Sparse Router (Work-Stealing)
- **`sparse_router.rs`** — RAW_ROUTE algorithm for tensor task scheduling
  - Memory-locality-aware work stealing
  - Region ownership tracking
  - Topology-aware routing decisions

- **`sparse_router_kani.rs`** — Formal verification of router invariants
  - 8 proven theorems about routing correctness
  - Kani proofs: all tasks routed, locality preserved, no deadlock, migration cost minimized

### Conformance Testing
- **`conformance_bridge.rs`** — Cross-language validation harness
  - Rust tests that exercise the FFI
  - Oracle functions for descriptor comparison
  - Both Rust and Pascal implementations must produce identical results

---

## Core Tensor Descriptor (ABI Contract)

```c
typedef struct {
    uint64_t buffer_id;         /* Unique buffer identity */
    uint64_t base_addr;         /* Physical address */
    uint64_t offset;            /* Offset into buffer */
    
    uint32_t rank;              /* Tensor rank (# dimensions) */
    uint64_t element_count;     /* Total elements */
    
    uint64_t shape_ptr;         /* Pointer to shape array */
    uint64_t stride_ptr;        /* Pointer to stride array */
    
    uint8_t governance;         /* B: 0=shared, 1=canonical */
    uint8_t ownership;          /* OWNED/SHARED/EXCLUSIVE */
    uint8_t materialized;       /* 0=not, 1=materialized */
    uint8_t state;              /* Internal state */
    
    uint64_t generation;        /* Generation counter (stale detection) */
} tensor_descriptor_t;
```

**Address Calculation (no framework overhead):**
```
address = base + offset + Σ(index[i] * stride[i])
```

---

## Ownership Semantics Through FFI

### Allocation
```
tensor_allocate(rank, shape, element_size)
  → TensorDescriptorC { buffer_id=N, generation=G, governance=1 (canonical) }
  → Arc<Vec<f32>> created internally
```

### View (Slice)
```
tensor_view(source, start_indices, end_indices)
  → TensorDescriptorC { buffer_id=M (new), generation=G' }
  → Arc::clone() increments refcount on source's data
  → shape/stride/offset computed, data NOT copied
  → is_contiguous recalculated
```

### Materialization
```
tensor_materialize(source where is_contiguous=false)
  → new Arc<Vec> allocated
  → strided data copied to new contiguous buffer
  → TensorDescriptorC { buffer_id=K (new), generation=G'', governance=1 }
  → source buffer remains alive and valid
  → ownership is exclusive on new buffer
```

### Deallocation
```
tensor_deallocate(descriptor)
  → Registry entry removed
  → Arc is dropped
  → if refcount reaches 0, buffer freed
```

---

## Generation Counter Safety

Every buffer has a **generation** counter that increments on:
- **Allocation** — generation = 1
- **Materialization** — new buffer gets new generation
- **Reallocation** — if buffer is reassigned

**Stale Descriptor Detection:**
```rust
fn tensor_validate_generation(desc: &TensorDescriptorC) -> Result {
    if registry[desc.buffer_id].generation != desc.generation {
        return Err("StaleGeneration");
    }
    Ok(())
}
```

Prevents use-after-free and races on descriptor validity.

---

## Governance State Machine

```
                SHARED (B=0)
                     │
              ┌──────┴──────┐
              │             │
          MATERIALIZE    VIEW/SLICE
              │             │
              ▼             ▼
         EXCLUSIVE      SHARED (B=0)
           (B=1)             │
             │               │
             └───────────────┘
```

**Invariants:**
- `B=0`: shared view, non-canonical strides permitted, materialization available
- `B=1`: exclusive owner, canonical strides, direct element access
- Generation increments on every transition
- Invalid transitions → rejected with fail-closed error

---

## Cross-Language Conformance

For every operation, both **Rust** and **Pascal** must produce:

1. **Identical shape** (rank and dimensions)
2. **Identical strides** (same canonical form)
3. **Identical offset**
4. **Identical element values** (same underlying data)
5. **Identical governance** (B value)
6. **Identical generation** (stale check)
7. **Identical buffer identity** (whether shared or independent)

**Oracle Function:**
```rust
fn conformance_compare_descriptors(
    rust: &TensorDescriptorC,
    pascal: &TensorDescriptorC
) -> ConformanceResult { ... }
```

If Rust and Pascal produce different descriptors, conformance fails.

---

## Sparse Router (RAW_ROUTE)

### Algorithm

```
route(task):
    r ← task.memory_region
    
    if RegionTable[r].owner exists:
        push(task, Worker[owner].deque)
        return owner
    
    candidates ← SparseTopology[r]
    best ← NONE
    best_score ← -∞
    
    for worker in candidates:
        score ← 0.35 * locality(worker, task)
              + 0.20 * memory_affinity(worker, task)
              + 0.25 * queue_pressure(worker)
              + 0.10 * stealability(worker)
              - 0.10 * migration_cost(worker, task)
        
        if score > best_score:
            best ← worker
            best_score ← score
    
    if best != NONE:
        push(task, Worker[best].deque)
        RegionTable[r].owner ← best
        return best
    
    worker ← select_least_loaded_worker()
    push(task, Worker[worker].deque)
    RegionTable[r].owner ← worker
    return worker
```

### Kani-Verified Theorems

1. **Progress**: Every task is routed ∀t: ∃w such that route(t) = w
2. **Locality**: Locality score dominates queue balance
3. **Region Consistency**: Once region R → worker W, reused until unset
4. **Migration Cost**: High-cost migrations are avoided over balance
5. **Fallback**: Empty topology always assigns via least-loaded fallback
6. **No Deadlock**: No circular region ownership
7. **Score Bounds**: Score ∈ [-1.0, 1.0]
8. **Stability**: Route decisions remain valid until topology changes

---

## Error Handling (Fail-Closed)

All FFI operations return `TensorResultC`:
```c
typedef struct {
    int32_t success;       /* 1 = OK, 0 = error */
    uint32_t error_code;   /* machine-readable code */
    char error_msg[256];   /* human-readable message */
} TensorResultC;
```

**Error codes:**
- `1`: Invalid pointer or rank
- `2`: Zero-element shape
- `3`: Buffer not found in registry
- `4`: Indices out of bounds
- `5`: Stale generation
- `6`: Computed address out of bounds
- `7`: Shared tensor cannot be mutated

On error: operation is rejected, no partial state is applied.

---

## Test Matrix

See `./tests/ffi/conformance_bridge.rs` for comprehensive coverage:

- `test_allocate_conforms` — Allocation produces canonical descriptor
- `test_view_preserves_data_and_buffer` — Views don't copy, source remains valid
- `test_materialization_creates_new_buffer` — New buffer, source unmodified
- `test_generation_safety` — Stale descriptors rejected
- `test_bounds_checking` — Out-of-bounds access fails
- `test_shared_ownership_prevents_store` — Arc refcount prevents mutation

Each test exercises both the Rust FFI and validates against Pascal oracle.

---

## Integration

The FFI layer enables:

1. **Hardware execution**: Tensor descriptors can be mapped directly to hardware registers/DMA
2. **Cross-language validation**: Pascal reference model verifies Rust correctness
3. **Formal reasoning**: Kani proves routing correctness, temporal safety, bounds
4. **Deterministic execution**: No implicit copies, ownership explicit, all state transitions auditable

The FFI is the **binary boundary** between Rust hardware semantics and Pascal reference semantics. Both must conform to the same ABI contract.
