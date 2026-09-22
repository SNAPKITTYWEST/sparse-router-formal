# NraayTensor Hardware Implementation Layer

**Rust hardware layer** for NraayTensor with explicit state transitions, generation-based safety, and fail-closed validation.

## Overview

This crate implements the core hardware tensor primitives for the NraayTensor system:
- **Buffer management** with generation counters and Arc-based ownership
- **Ownership state machine** (B_layout, B_own) with explicit transitions
- **Memory address calculation** with stride handling for multi-dimensional indexing
- **Tensor views and slicing** (zero-copy Arc::clone, never copy data)
- **Materialization barrier** (transactional copy-on-write with source preservation)
- **Hardware-safe C-compatible descriptors** (#[repr(C)] for FFI)
- **Governance state engine** with fail-closed validation gates
- **Sparse work-stealing scheduler** (RAW_ROUTE with memory locality heuristic)

## Core Principles

### 1. Explicit State Transitions
Every operation transitions ownership and governance state through a documented state machine:
- VIEW → (shared, canonical or non-canonical)
- SLICE → (shared, non-canonical)
- MATERIALIZE → (exclusive, canonical) with new buffer
- TRANSPOSE → (non-canonical strides)

### 2. Generation Counter Safety
Each buffer has a generation counter that increments on allocation/materialization:
- Prevents use-after-free via stale descriptor detection
- Every view holds buffer_id + generation
- Operations validate generation match before access

### 3. Fail-Closed Validation
All operations validate preconditions before modifying state:
- Bounds checking on all indices
- Mutability checks (exclusive owner only)
- Stale generation rejection
- No partial state application on error

### 4. No Hidden Ownership Changes
All ownership transitions are explicit:
- VIEW always returns shared (B_own: 1 → 0)
- Arc::clone is transparent (no hidden increments)
- Refcount checked to enforce exclusivity

## Architecture

### Modules

| Module | Purpose | LOC |
|--------|---------|-----|
| **buffer.rs** | Buffer allocation, generation counters, Arc ownership | 180 |
| **ownership.rs** | State machine (B_layout, B_own) with transitions | 240 |
| **memory.rs** | Address calculation, stride handling, MemoryLayout | 310 |
| **view.rs** | Tensor views, slicing, transpose, reshape (zero-copy) | 320 |
| **materialize.rs** | Transactional materialization (read→alloc→copy→install) | 290 |
| **descriptor.rs** | C-compatible #[repr(C)] TensorDescriptor | 280 |
| **governance.rs** | Validation gates for all operations | 310 |
| **routing.rs** | RAW_ROUTE sparse work-stealing scheduler | 380 |
| **error.rs** | Fail-closed error types | 50 |
| **lib.rs** | Core TensorHardware API & integration | 200 |
| **Total** | | **2,560** |

## API Reference

### Tensor Hardware Context

```rust
pub struct TensorHardware {
    validator: GovernanceValidator,
    materializer: MaterializationBarrier,
    router: SparseRouter,
}

impl TensorHardware {
    pub fn allocate(&mut self, shape: Vec<usize>) -> Result<(Buffer, TensorView)>;
    pub fn create_view(&mut self, view: &TensorView, buffer: &Buffer) -> Result<TensorView>;
    pub fn slice(&mut self, view: &TensorView, buffer: &Buffer, 
                 ranges: &[(usize, usize, usize)]) -> Result<TensorView>;
    pub fn transpose(&mut self, view: &TensorView, buffer: &Buffer, 
                     permutation: &[usize]) -> Result<TensorView>;
    pub fn reshape(&mut self, view: &TensorView, buffer: &Buffer, 
                   new_shape: Vec<usize>) -> Result<TensorView>;
    pub fn materialize(&mut self, view: &TensorView, 
                       buffer: &Buffer) -> Result<MaterializationResult>;
    pub fn get(&mut self, view: &TensorView, buffer: &Buffer, 
               indices: &[usize]) -> Result<f32>;
    pub fn set(&mut self, view: &TensorView, buffer: &Buffer, 
               indices: &[usize], value: f32) -> Result<()>;
    pub fn release(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()>;
}
```

### Buffer Management

```rust
pub struct Buffer {
    data: Arc<Vec<f32>>,
    generation: u64,
    id: u64,
    is_owner: bool,
}

impl Buffer {
    pub fn allocate(id: u64, size: usize) -> Result<Self>;
    pub fn strong_count(&self) -> usize;
    pub fn is_exclusive(&self) -> bool;
    pub fn validate_generation(&self, gen: u64) -> Result<()>;
    pub fn check_mutable(&self) -> Result<()>;
}
```

### Ownership State Machine

```rust
pub struct OwnershipToken {
    pub is_owner: bool,      // B_own: 1 = exclusive, 0 = shared
    pub is_canonical: bool,  // B_layout: 1 = canonical, 0 = non-canonical
    pub version: u32,        // State transition counter
}

impl OwnershipToken {
    pub fn new_exclusive() -> Self;
    pub fn transition_to_shared(&self) -> Result<Self>;
    pub fn transition_to_non_canonical(&self) -> Result<Self>;
    pub fn transition_to_materialized(&self) -> Result<Self>;
    pub fn can_mutate_directly(&self) -> bool;
    pub fn needs_materialization(&self) -> bool;
}
```

### Tensor Views

```rust
pub struct TensorView {
    buffer_handle: BufferHandle,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    ownership: OwnershipToken,
}

impl TensorView {
    pub fn from_buffer(buffer: &Buffer, layout: MemoryLayout) -> Self;
    pub fn share(&self) -> Result<Self>;
    pub fn slice(&self, ranges: &[(usize, usize, usize)]) -> Result<Self>;
    pub fn transpose(&self, permutation: &[usize]) -> Result<Self>;
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self>;
    pub fn address_for(&self, indices: &[usize]) -> Result<usize>;
}
```

### Materialization

```rust
pub struct MaterializationBarrier;

impl MaterializationBarrier {
    pub fn materialize(&mut self, view: &TensorView, 
                       buffer: &Buffer) -> Result<MaterializationResult>;
}

pub struct MaterializationResult {
    pub new_buffer: Buffer,
    pub new_view: TensorView,
    pub source_buffer_id: u64,
}
```

### Hardware Descriptor

```rust
#[repr(C)]
pub struct TensorDescriptor {
    pub buffer_id: u64,
    pub offset: u64,
    pub rank: u32,
    pub element_count: u64,
    pub shape_ptr: u64,
    pub stride_ptr: u64,
    pub ownership: u8,      // OwnershipCode
    pub governance: u8,     // GovernanceCode
    pub materialization: u8, // MaterializationCode
    pub generation: u64,
}
```

### Governance Validation

```rust
pub struct GovernanceValidator;

impl GovernanceValidator {
    pub fn validate_view(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()>;
    pub fn validate_access(&mut self, view: &TensorView, buffer: &Buffer, 
                          indices: &[usize], is_write: bool) -> Result<()>;
    pub fn validate_slice(&mut self, view: &TensorView, 
                         ranges: &[(usize, usize, usize)]) -> Result<()>;
    pub fn validate_materialize(&mut self, view: &TensorView, 
                                buffer: &Buffer) -> Result<()>;
}
```

### Work-Stealing Router

```rust
pub struct SparseRouter;

impl SparseRouter {
    pub fn new() -> Self;
    pub fn register_worker(&mut self, id: WorkerId, locality_distance: f32);
    pub fn route(&mut self, task: &Task) -> WorkerId;
    pub fn region_owner(&self, region: RegionId) -> Option<WorkerId>;
}

pub struct Task {
    pub id: u64,
    pub region: RegionId,
    pub size: usize,
}
```

## Usage Examples

### Allocate and Access Elements

```rust
let mut hw = TensorHardware::new();

// Allocate exclusive, canonical 10x10 tensor
let (buffer, view) = hw.allocate(vec![10, 10])?;

// Set element (requires exclusive ownership)
hw.set(&view, &buffer, &[2, 3], 42.0)?;

// Get element (read-only, allowed on shared too)
let val = hw.get(&view, &buffer, &[2, 3])?;
assert_eq!(val, 42.0);
```

### Create Shared Views (Zero-Copy)

```rust
// Create shared view (Arc::clone, no data copy)
let shared = hw.create_view(&view, &buffer)?;

// Can read from shared
let val = hw.get(&shared, &buffer, &[2, 3])?;

// Cannot write to shared
hw.set(&shared, &buffer, &[2, 3], 50.0)?; // ❌ Err(SharedTensorMutation)
```

### Slicing (Non-Destructive)

```rust
// Slice [2:5, 3:7] (creates shared view with new shape)
let sliced = hw.slice(&view, &buffer, &[(2, 5, 1), (3, 7, 1)])?;

// New shape, same data
assert_eq!(sliced.shape, vec![3, 4]);

// Source tensor unchanged
assert!(view.is_exclusive());
```

### Transpose (Non-Canonical Strides)

```rust
// Transpose swaps dimensions → non-canonical strides
let transposed = hw.transpose(&view, &buffer, &[1, 0])?;

// Cannot directly access (non-canonical)
assert!(!transposed.is_contiguous());

// Needs materialization to make canonical
```

### Materialization (Transactional Copy-On-Write)

```rust
// Materialize non-canonical to contiguous buffer
let result = hw.materialize(&transposed, &buffer)?;

// New buffer is exclusive and canonical
assert!(result.new_view.is_exclusive());
assert!(result.new_view.is_contiguous());

// Original buffer still valid
assert!(buffer.is_exclusive());
```

### Governance Validation

```rust
let mut validator = GovernanceValidator::new();

// Validate view against buffer
validator.validate_view(&view, &buffer)?;

// Validate element access
validator.validate_access(&view, &buffer, &[0, 0], true)?;

// Get operation sequence
let ops = validator.sequence();
```

### Work-Stealing Scheduling

```rust
let mut router = SparseRouter::new();

// Register workers
router.register_worker(WorkerId(0), 0.5);  // close
router.register_worker(WorkerId(1), 1.5);  // far

// Route task to best worker
let task = Task {
    id: 1,
    region: RegionId(100),
    size: 1024,
};

let worker = router.route(&task);

// Same region tasks go to same worker (locality)
let task2 = Task { id: 2, region: RegionId(100), size: 512 };
assert_eq!(router.route(&task2), worker);
```

## Testing

All 49 unit tests pass:

```bash
cargo test --lib
```

**Test coverage:**
- Buffer allocation and generation
- Ownership state transitions
- Memory address calculation
- View creation and slicing
- Materialization workflow
- Governance validation
- Error handling (fail-closed)
- Routing and scheduling

## Performance Characteristics

- **Allocation**: O(n) where n = element count
- **View/Slice**: O(rank) - no data copy
- **Address calculation**: O(rank)
- **Materialize**: O(n) - single sequential pass
- **Element access**: O(1) after bounds validation

## Safety Model

1. **Generation Counter**: Stale descriptor detection
2. **Arc Reference Counting**: Exclusive ownership enforcement
3. **Bounds Checking**: All indices validated before access
4. **Fail-Closed Validation**: No partial state on error
5. **Type Safety**: Rust's type system + explicit ownership tokens

## FFI Compatibility

The descriptor (#[repr(C)]) is compatible with:
- Pascal (Free Pascal FFI)
- C/C++ bindings
- Hardware interfaces (DMA, registers)

Size: 72 bytes (64-bit aligned)

## Dependencies

- `thiserror` — Error type derivation
- `parking_lot` — Parking lot mutex (optional)
- `atomic` — Atomic types

## License

GPL-3.0-or-later OR Apache-2.0 (dual licensed)

## References

- **Architecture**: Design Phase NraayTensor specification
- **Formal Verification**: Kani proofs in `ffi/sparse_router_kani.rs`
- **FFI Contract**: `FFI_CONTRACT.md`
- **Example Usage**: `examples/core_operations.rs`
