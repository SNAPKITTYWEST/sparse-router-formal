# MEMORY_MODEL.md
## Physical Storage, Address Generation, and Memory Domains

### 1. Overview

The memory model specifies:
- **Physical Storage (𝒫)**: Layout of actual tensor data in heap memory
- **Semantic View (𝒮)**: Logical interpretation via (shape, strides, offset)
- **Index Mapping Formula**: Conversion from logical coordinates to physical addresses
- **Memory Domains**: Classification of storage by access patterns and safety properties
- **C-Compatible Layout**: Binary representation for FFI and Pascal bridging

---

### 2. Physical Storage (𝒫)

**Definition**: A strictly contiguous 1D array of f32 elements in heap memory.

```
𝒫 = [elem₀, elem₁, elem₂, ..., elem_{N-1}]
where N = ∏ shape_i (for the logical view)
```

**Rust Representation**:
```rust
pub struct NraayTensor {
    data: Arc<Vec<f32>>,    // ← 𝒫 lives here
    // ... metadata ...
}

// Access pattern:
let phys_index: usize = offset + Σ(coord[i] * strides[i]);
let value: f32 = data[phys_index];
```

**Allocation**:
```rust
// Creation allocates:
data = Arc::new(vec![0.0; total_size])

// Size = ∏ shape_i (for the view at creation time)
```

**Deallocation**:
```rust
// Automatic when last Arc reference drops
// RefCount management via Arc::strong_count
```

---

### 3. Semantic View (𝒮)

**Definition**: Tuple (S, O, Σ) where:
- **S (Shape)**: Logical dimensions (s₀, s₁, ..., s_{d-1})
- **O (Offset)**: Starting physical index in 𝒫
- **Σ (Strides)**: Step sizes (σ₀, σ₁, ..., σ_{d-1})

**Example**:

```
Physical Buffer 𝒫: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
                     0  1  2  3  4  5  6  7  8  9  10  11

View 1: Shape=[2,3], Offset=0, Strides=[3,1]
  Logical (0,0) → Physical 0
  Logical (0,1) → Physical 1
  Logical (0,2) → Physical 2
  Logical (1,0) → Physical 3
  Logical (1,1) → Physical 4
  Logical (1,2) → Physical 5

View 2: Shape=[3,2], Offset=1, Strides=[4,1]
  (after TRANSPOSE and SLICE)
  Logical (0,0) → Physical 1
  Logical (0,1) → Physical 2
  Logical (1,0) → Physical 5
  Logical (1,1) → Physical 6
  Logical (2,0) → Physical 9
  Logical (2,1) → Physical 10
```

---

### 4. Index Mapping Formula

**Core Formula**:
```
For logical coordinate X = (x₀, x₁, ..., x_{d-1}):

physical_address(X) = O + Σᵢ (xᵢ * σᵢ)
                    = offset + (x₀ * stride₀ + x₁ * stride₁ + ... + x_{d-1} * stride_{d-1})
```

**Preconditions**:
- shape.len() = strides.len() = coordinate.len()
- ∀ i: 0 ≤ xᵢ < sᵢ
- physical_address(X) < |𝒫|

**Failure Conditions**:
- coordinate.len() ≠ shape.len() → DimensionMismatch
- ∃ i: xᵢ ≥ sᵢ → IndexOutOfBounds (logical)
- physical_address(X) ≥ |𝒫| → IndexOutOfBounds (physical)

**Pseudocode**:

```rust
fn compute_physical_address(
    offset: usize,
    coord: &[usize],
    strides: &[usize],
) -> Result<usize> {
    if coord.len() != strides.len() {
        return Err(DimensionMismatch);
    }
    
    let mut addr = offset;
    for i in 0..coord.len() {
        addr += coord[i] * strides[i];
    }
    Ok(addr)
}
```

**Edge Cases**:

```
Scalar (rank-0):
  Shape=[], Strides=[], Offset=addr
  physical_address() = offset (no summation)

Vector (rank-1):
  Shape=[n], Strides=[1], Offset=0
  physical_address([i]) = i (standard dense)

Vector (rank-1, strided):
  Shape=[n], Strides=[s], Offset=0
  physical_address([i]) = i * s (every s-th element)

Matrix (rank-2, canonical):
  Shape=[m,n], Strides=[n,1], Offset=0
  physical_address([i,j]) = i*n + j (row-major)

Matrix (rank-2, transposed):
  Shape=[n,m], Strides=[1,n], Offset=0
  physical_address([i,j]) = i + j*n (column-major)
```

---

### 5. Canonical (Row-Major) Strides

**Definition**:
```
For shape (s₀, s₁, ..., s_{d-1}), canonical strides are:
  σᵢ = ∏_{j>i} sⱼ

i.e., σᵢ = s_{i+1} * s_{i+2} * ... * s_{d-1}
```

**Formula**:
```rust
fn canonical_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}
```

**Examples**:

```
Shape [2, 3]:
  σ₀ = 3 * 1 = 3
  σ₁ = 1
  Canonical strides = [3, 1]

Shape [2, 3, 4]:
  σ₀ = 3 * 4 = 12
  σ₁ = 4
  σ₂ = 1
  Canonical strides = [12, 4, 1]

Shape [5]:
  σ₀ = 1
  Canonical strides = [1]
```

**B_layout Flag**:
```
B_layout = 1  ⟺  strides == canonical_strides(shape)
B_layout = 0  ⟺  strides != canonical_strides(shape)
```

---

### 6. TensorDescriptor C-Compatible Layout

**Binary Representation** (for FFI and Pascal bridge):

```
Offset   Size (bytes)   Field                Type
0        8              data_ptr             *void (Arc wrapper stripped)
8        8              data_len             u64
16       8              shape_ptr            *u64
24       8              shape_len            u64
32       8              strides_ptr          *u64
40       8              strides_len          u64
48       8              offset                u64
56       1              is_contiguous        u8 (bool)
57       1              owns_storage         u8 (bool)
58       2              _padding             u16 (for alignment)
60       8              generation           u64
68       8              refcount             u64

Total:   76 bytes (aligned to 8-byte boundary)
```

**Rust Structure** (with C repr):

```rust
#[repr(C)]
pub struct TensorDescriptor {
    pub data_ptr: *const f32,      // Points to Vec<f32> data
    pub data_len: u64,             // len() of Vec<f32>
    pub shape_ptr: *const u64,     // Points to shape array
    pub shape_len: u64,            // Number of dimensions
    pub strides_ptr: *const u64,   // Points to strides array
    pub strides_len: u64,          // Number of dimensions
    pub offset: u64,               // Logical offset
    pub is_contiguous: u8,         // B_layout flag
    pub owns_storage: u8,          // B_own flag
    _padding: u16,
    pub generation: u64,           // Generation counter
    pub refcount: u64,             // Arc strong count (snapshot)
}

impl TensorDescriptor {
    pub fn from_tensor(t: &NraayTensor) -> Self {
        TensorDescriptor {
            data_ptr: t.data.as_ptr(),
            data_len: t.data.len() as u64,
            shape_ptr: t.shape.as_ptr() as *const u64,
            shape_len: t.shape.len() as u64,
            strides_ptr: t.strides.as_ptr() as *const u64,
            strides_len: t.strides.len() as u64,
            offset: t.offset as u64,
            is_contiguous: if t.is_contiguous { 1 } else { 0 },
            owns_storage: if t.owns_storage { 1 } else { 0 },
            _padding: 0,
            generation: t.generation,
            refcount: Arc::strong_count(&t.data) as u64,
        }
    }
}
```

**Pascal Equivalent**:

```pascal
type
  TensorDescriptor = record
    data_ptr: Pointer;           // Points to f32 array
    data_len: UInt64;
    shape_ptr: ^UInt64;
    shape_len: UInt64;
    strides_ptr: ^UInt64;
    strides_len: UInt64;
    offset: UInt64;
    is_contiguous: Byte;         // B_layout
    owns_storage: Byte;          // B_own
    padding: Word;
    generation: UInt64;
    refcount: UInt64;
  end;
```

---

### 7. Memory Domain Classification

**Definition**: Categorization of storage by access patterns and safety properties.

#### Domain 0: Register (Ultra-fast, no materialization)
- Immediate scalar values
- Single-element tensors
- Optimization: keep in CPU register if possible

#### Domain 1: Local (Fast, heap-like but short-lived)
- Temporary tensors created and freed within function scope
- Materialization cost is low (used once)

#### Domain 2: Shared (Medium-speed, COW-protected)
- Multiple views sharing single buffer
- COW barrier isolates mutations
- Refcount > 1

#### Domain 3: Numa (Slow, multi-socket)
- Large tensors spanning NUMA nodes
- Address translation adds latency
- Materialization may trigger rebalancing

#### Domain 4: Host (Normal heap, CPU-accessible)
- Standard host memory (default)
- No special access patterns

#### Domain 5: Device (GPU/Accelerator memory)
- Off-CPU storage (e.g., GPU VRAM)
- Transfer cost significant
- May require explicit synchronization

#### Domain 6: MMIO (Memory-mapped I/O)
- Tensors backed by memory-mapped files or hardware registers
- Ultra-low latency but single-threaded access
- Coordination required

**Runtime Domain Selection**:

```rust
fn classify_domain(tensor: &NraayTensor) -> MemoryDomain {
    if tensor.len() == 1 {
        MemoryDomain::Register
    } else if tensor.refcount() > 1 {
        MemoryDomain::Shared
    } else if tensor.data.len() > 1_000_000 {
        // Heuristic: large tensors may be in NUMA
        MemoryDomain::Numa
    } else {
        MemoryDomain::Host
    }
}
```

---

### 8. Stride/Offset Computation Without Framework Overhead

**Goal**: Minimize computation cost for address generation (single addition + multiplication chain).

**Optimized Pseudocode**:

```rust
#[inline]
fn get_element(tensor: &NraayTensor, coord: &[usize]) -> Result<f32> {
    debug_assert_eq!(coord.len(), tensor.shape.len());
    
    // Single pass: compute address as sum of products
    let mut addr = tensor.offset;
    for i in 0..coord.len() {
        debug_assert!(coord[i] < tensor.shape[i]);
        addr += coord[i] * tensor.strides[i];
    }
    
    debug_assert!(addr < tensor.data.len());
    Ok(tensor.data[addr])
}
```

**Complexity**:
- Time: O(d) where d = rank (number of dimensions)
- Space: O(1) (only local variables)
- No allocations, no Arc operations

**Macro-Optimization (for hot paths)**:

```rust
// Specialized for rank-2 (matrix) operations
#[inline]
fn get_matrix_element_fast(
    tensor: &NraayTensor,
    i: usize,
    j: usize,
) -> f32 {
    let addr = tensor.offset + i * tensor.strides[0] + j * tensor.strides[1];
    tensor.data[addr]
}
```

---

### 9. Allocation Patterns

**Pattern 1: Contiguous Dense Tensor**

```
Buffer Layout:
[elem_0,0  elem_0,1  ...  elem_0,n-1  |  elem_1,0  ...  elem_m-1,n-1]
 ←─────────────────────────────────────→  ←─────────────────────────→
    Row 0 (contiguous)                       Other rows

Strides: [n, 1]
Offset: 0
B_layout: 1
```

**Pattern 2: Strided View (Every k-th element)**

```
Buffer Layout:
[elem_0  skip  skip  elem_k  skip  skip  elem_2k  ...]
 ↑                    ↑                    ↑
 └────────────────────┴────────────────────┘
    Stride = k

Strides: [k]
Offset: 0
B_layout: 0 (non-canonical)
```

**Pattern 3: Sliced View (Offset Start)**

```
Buffer Layout:
[skip  skip  elem_start  elem_start+1  ...  elem_end-1  skip]
                ↑
                │
             Offset = start

Strides: [1]
Offset: start
B_layout: 1 (canonical, but offset ≠ 0)
```

**Pattern 4: Transposed View (Permuted Axes)**

```
Original: Shape=[2,3], Strides=[3,1]
  (0,0)→0  (0,1)→1  (0,2)→2
  (1,0)→3  (1,1)→4  (1,2)→5

Transposed: Shape=[3,2], Strides=[1,3]
  (0,0)→0  (0,1)→3
  (1,0)→1  (1,1)→4
  (2,0)→2  (2,1)→5

Strides: [1, 3]
Offset: 0
B_layout: 0 (non-canonical)
```

---

### 10. Materialization Memory Effects

**Before Materialization**:
```
Original tensor:
  data → Arc @ addr_A pointing to [elem_0, ..., elem_{N-1}]
  shape = [3, 2]
  strides = [1, 3]      (non-canonical)
  offset = 0
  B_layout = 0
  B_own = 1 (exclusive)

Viewed access pattern:
  logical(0,0) → physical(0)
  logical(0,1) → physical(3)
  logical(1,0) → physical(1)
  logical(1,1) → physical(4)
  logical(2,0) → physical(2)
  logical(2,1) → physical(5)
```

**After Materialization**:
```
Materialized tensor:
  data → Arc @ addr_B pointing to [elem_0, elem_3, elem_1, elem_4, elem_2, elem_5]
  shape = [3, 2]        (unchanged)
  strides = [2, 1]      (canonical for new shape)
  offset = 0
  B_layout = 1
  B_own = 1

New access pattern (canonical):
  logical(0,0) → physical(0)
  logical(0,1) → physical(1)
  logical(1,0) → physical(2)
  logical(1,1) → physical(3)
  logical(2,0) → physical(4)
  logical(2,1) → physical(5)

Old buffer @ addr_A: still valid (not deallocated)
```

---

### 11. References

- Tensor Contract: TENSOR_CONTRACT.md (operations affecting memory)
- Governance State Machine: GOVERNANCE_STATE_MACHINE.md (state transitions)
- Temporal Safety: TEMPORAL_SAFETY.md (buffer lifetime guarantees)
- Proof Obligations: PROOF_OBLIGATIONS.md (PO-1, PO-2, PO-4 address validation)
