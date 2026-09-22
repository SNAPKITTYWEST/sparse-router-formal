# TENSOR_CONTRACT.md
## Normative Operation Signatures for NraayTensor

### 1. Overview

The NraayTensor Contract defines the complete interface for all tensor operations. Each operation specifies:
- **Input State**: Pre-conditions on tensor governance flags (B_layout, B_own), shape, and ownership
- **Output State**: Post-conditions on governance flags and memory configuration
- **Ownership Effects**: Changes to Arc refcount and exclusive-access semantics
- **Memory Effects**: Effects on physical buffer 𝒫, offset O, strides Σ
- **Failure Conditions**: Explicit rejection criteria (bounds, dimension mismatch, invalid state)

All operations are **fail-closed**: when preconditions fail, the tensor remains unchanged and an error is returned.

---

### 2. Data Structures

#### TensorDescriptor (C-Compatible Layout)

```rust
pub struct NraayTensor {
    // Ownership token (temporal safety via Arc<Vec<f32>>)
    data: Arc<Vec<f32>>,
    
    // Semantic view metadata
    shape: Vec<usize>,        // S = (s_0, ..., s_{d-1})
    strides: Vec<usize>,      // Σ = (σ_0, ..., σ_{d-1})
    offset: usize,            // O (memory offset)
    
    // Binary governance flags
    is_contiguous: bool,      // B_layout = 1 ⟺ canonical strides
    owns_storage: bool,       // B_own = 1 ⟺ refcount = 1
}
```

**Invariant**: The physical buffer 𝒫 has length ≥ max accessed address in any view.

---

### 3. Operation Specifications

#### OP-1: ALLOC (Constructor)

**Signature**:
```
ALLOC(shape: Vec<usize>) -> Result<NraayTensor>
```

**Precondition**:
- shape is non-empty
- ∀ s_i ∈ shape: s_i > 0
- ∏ s_i is finite and allocatable

**Input State**:
- N/A (construction)

**Output State**:
```
shape'        = shape
strides'      = canonical_strides(shape)
offset'       = 0
B_layout'     = 1  (newly allocated buffer is contiguous)
B_own'        = 1  (exclusive ownership, refcount = 1)
buffer'       = Arc::new(vec![0.0; size])
```

**Ownership Effect**:
- refcount = 1 (constructor holds sole Arc)

**Memory Effect**:
- Allocates new 1D buffer in heap
- Physical storage 𝒫 = [0.0; ∏ s_i]

**Failure Conditions**:
```
shape.is_empty()                   → EmptyTensor
any s_i = 0                        → EmptyTensor
∏ s_i causes overflow              → OutOfMemory
allocation fails                   → OutOfMemory
```

**Oracle Behavior** (Rust & Pascal must match):
```
let t = ALLOC([2, 3])
assert(t.shape = [2, 3])
assert(t.strides = [3, 1])
assert(t.offset = 0)
assert(t.B_layout = true)
assert(t.B_own = true)
assert(t.refcount() = 1)
```

---

#### OP-2: FREE (Destructor)

**Signature**:
```
FREE(tensor: &mut NraayTensor)
```

**Precondition**:
- tensor is valid (within scope)

**Input State**:
- Any governance configuration allowed

**Output State**:
- Tensor deallocated
- If refcount reaches 0, buffer 𝒫 deallocated
- Otherwise, Arc refcount decremented by 1

**Ownership Effect**:
- Arc::drop() called on self.data
- refcount decrements by 1
- When refcount = 0, Vec<f32> deallocated by Rust allocator

**Memory Effect**:
- Heap memory freed only if this was the last owner (refcount was 1)

**Failure Conditions**: None

**Oracle Behavior**:
```
let t = ALLOC([2, 3])            // refcount = 1
let s = SLICE(t, ...)            // refcount = 2
DROP(s)                          // refcount = 1
DROP(t)                          // refcount = 0 → heap freed
```

---

#### OP-3: VIEW (Slicing)

**Signature**:
```
VIEW(self: &NraayTensor, ranges: &[(start, end, step)]) 
  -> Result<NraayTensor>
```

**Precondition**:
- ranges.len() = self.shape.len()
- ∀ i: 0 ≤ start_i < end_i ≤ shape_i
- ∀ i: step_i > 0

**Input State**:
```
self.B_layout ∈ {0, 1}    (any contiguity)
self.B_own ∈ {0, 1}       (any ownership)
```

**Output State**:
```
shape'_i      = ⌈(end_i - start_i) / step_i⌉
strides'_i    = strides_i * step_i
offset'       = offset + Σ_i (start_i * strides_i)
B_layout'     = recompute_canonical(shape', strides')
B_own'        = 0          (always shared after slicing)
data'         = Arc::clone(&self.data)
```

**Ownership Effect**:
- refcount increments by 1 (Arc clone)
- New tensor acquires non-exclusive view (owns_storage = false)

**Memory Effect**:
- Zero-copy: no data is copied
- Offset computation may skip leading elements
- Stride multiplication may create non-canonical access patterns

**Failure Conditions**:
```
ranges.len() ≠ shape.len()        → DimensionMismatch
start_i ≥ end_i                   → InvalidSliceRange
step_i = 0                        → InvalidSliceRange
start_i >= shape_i                → InvalidSliceRange
end_i > shape_i                   → InvalidSliceRange
```

**Oracle Behavior**:
```
let t = ALLOC([4, 4])
t.set([0,0], 1.0); t.set([0,1], 2.0); ... t.set([3,3], 16.0)
let v = VIEW(t, [(0,2,1), (1,4,1)])
assert(v.shape = [2, 3])
assert(v.offset = 1)
assert(v.strides = [4, 1])
assert(v.B_own = false)
assert(v.refcount() = 2)
```

---

#### OP-4: TRANSPOSE (Logical Shift)

**Signature**:
```
TRANSPOSE(self: &mut NraayTensor, perm: &[usize]) -> Result<()>
```

**Precondition**:
- perm is a valid permutation of (0..self.shape.len())
- perm.len() = self.shape.len()

**Input State**:
```
self.B_layout ∈ {0, 1}
self.B_own ∈ {0, 1}
```

**Output State**:
```
shape'_i      = shape[perm[i]]
strides'_i    = strides[perm[i]]
offset'       = offset (unchanged)
B_layout'     = recompute_canonical(shape', strides')
B_own'        = owns_storage  (unchanged)
data'         = data           (unchanged, no copy)
```

**Ownership Effect**: None (metadata-only operation)

**Memory Effect**:
- Pure logical transformation of view metadata
- No data movement
- Physical buffer 𝒫 untouched

**Failure Conditions**:
```
perm.len() ≠ shape.len()          → InvalidPermutation
perm contains duplicates          → InvalidPermutation
∃ p ∈ perm: p >= shape.len()      → InvalidPermutation
perm is not a permutation         → InvalidPermutation
```

**Oracle Behavior**:
```
let mut t = ALLOC([2, 3])
assert(t.B_layout = true)
TRANSPOSE(&mut t, [1, 0])
assert(t.shape = [3, 2])
assert(t.B_layout = false)        // Non-canonical now
```

---

#### OP-5: RESHAPE (Safe Shape Change)

**Signature**:
```
RESHAPE(self: &mut NraayTensor, new_shape: Vec<usize>) 
  -> Result<()>
```

**Precondition**:
- new_shape is non-empty
- ∏ new_shape = ∏ self.shape (size preserved)
- B_layout = 1 AND B_own = 1 (only on contiguous exclusive tensors)

**Input State**:
```
B_layout = 1
B_own = 1
∏ shape = ∏ new_shape
```

**Output State**:
```
shape'        = new_shape
strides'      = canonical_strides(new_shape)
offset'       = 0
B_layout'     = 1
B_own'        = 1
data'         = data (unchanged)
```

**Ownership Effect**: None

**Memory Effect**:
- No allocation or copying
- Reinterprets physical buffer under new shape
- Offset reset to 0 (reshape only on contiguous exclusive)

**Failure Conditions**:
```
∏ new_shape ≠ ∏ shape              → ShapeMismatch
new_shape.is_empty()              → EmptyTensor
B_layout ≠ 1                       → NotContiguous
B_own ≠ 1                          → NotExclusive
```

**Oracle Behavior**:
```
let mut t = ALLOC([2, 6])
assert(t.len() = 12)
RESHAPE(&mut t, [3, 4])?
assert(t.shape = [3, 4])
assert(t.len() = 12)
```

---

#### OP-6: MATERIALIZE (Restore Contiguity + Exclusive Ownership)

**Signature**:
```
MATERIALIZE(self: &mut NraayTensor) -> Result<()>
```

**Precondition**: None (always safe)

**Input State**:
```
B_layout ∈ {0, 1}
B_own ∈ {0, 1}
Any shape/stride/offset valid
```

**Output State**:
```
IF B_layout = 1 AND B_own = 1:
  // No-op (already canonical and exclusive)
  output = input unchanged

IF B_layout = 0 OR B_own = 0:
  // Allocate, copy, and restore canonical state
  new_buffer = Arc::new(vec![0.0; len])
  copy_viewed_elements(self, new_buffer)
  
  shape'        = shape (unchanged)
  strides'      = canonical_strides(shape)
  offset'       = 0
  B_layout'     = 1
  B_own'        = 1
  data'         = Arc::new(new_buffer)
```

**Ownership Effect**:
- Decrements refcount on old data Arc by 1
- Creates new Arc with refcount = 1
- Old Arc unchanged for other views

**Memory Effect**:
- If precondition true: COW semantics apply
- New contiguous buffer allocated with element count = ∏ shape
- Elements copied in row-major logical order from source view
- Original buffer remains valid for other views

**Failure Conditions**:
```
copy_viewed_elements overflow      → IndexOutOfBounds
new allocation fails               → OutOfMemory
```

**Oracle Behavior**:
```
let mut t = ALLOC([2, 3])
TRANSPOSE(&mut t, [1, 0])
assert(t.B_layout = false)
MATERIALIZE(&mut t)?
assert(t.B_layout = true)
assert(t.B_own = true)
assert(t.strides = [2, 1])
assert(t.offset = 0)
```

---

#### OP-7: GET (Element Access - Read)

**Signature**:
```
GET(self: &NraayTensor, coord: &[usize]) -> Result<f32>
```

**Precondition**:
- coord.len() = self.shape.len()
- ∀ i: 0 ≤ coord_i < shape_i

**Input State**:
- Any governance state (read-only)

**Output State**:
- Unchanged

**Ownership Effect**: None

**Memory Effect**:
- Reads single element from buffer 𝒫
- Computes physical index: idx = offset + Σ (coord_i * strides_i)
- Bounds check: idx < |𝒫|

**Failure Conditions**:
```
coord.len() ≠ shape.len()          → DimensionMismatch
coord_i >= shape_i                 → IndexOutOfBounds
computed_idx >= buffer.len()       → IndexOutOfBounds
```

**Oracle Behavior**:
```
let mut t = ALLOC([2, 3])
SET(t, [0,1], 42.0)
assert(GET(t, [0,1]) = 42.0)
```

---

#### OP-8: SET (Element Access - Write)

**Signature**:
```
SET(self: &mut NraayTensor, coord: &[usize], value: f32) 
  -> Result<()>
```

**Precondition**:
- coord.len() = self.shape.len()
- ∀ i: 0 ≤ coord_i < shape_i

**Input State**:
```
B_own ∈ {0, 1}
```

**Output State**:
```
IF refcount(data) = 1:
  // Exclusive; direct mutation
  B_own'  = true
  B_layout' unchanged
  element_at(offset + Σ coord_i * strides_i) = value

IF refcount(data) > 1:
  // Shared; COW
  data'   = Arc::new(clone(data))
  B_own'  = true
  B_layout' unchanged
  element_at(0 + Σ coord_i * strides_i) = value
```

**Ownership Effect**:
- If refcount > 1: Arc refcount decrements by 1 on old buffer, new Arc created with refcount = 1
- If refcount = 1: no Arc change, in-place mutation via Arc::get_mut

**Memory Effect**:
- Single element written to buffer
- If COW triggered, full copy of data to new allocation before write
- Old buffer remains accessible to other views

**Failure Conditions**:
```
coord.len() ≠ shape.len()          → DimensionMismatch
coord_i >= shape_i                 → IndexOutOfBounds
computed_idx >= buffer.len()       → IndexOutOfBounds
```

**Oracle Behavior**:
```
let mut t = ALLOC([2, 3])
let s = VIEW(t, ...)               // refcount = 2
SET(s, [0,0], 99.0)               // COW triggered
assert(refcount(s) = 1)
assert(GET(t, [0,0]) = 0.0)       // Original unchanged
assert(GET(s, [0,0]) = 99.0)      // Slice modified
```

---

### 4. Invariants (Must Hold After Every Operation)

1. **Physical Buffer Invariant**: |𝒫| ≥ max accessed address
2. **Governance Correctness**: B_layout ⟺ strides canonical for shape
3. **Shape-Stride Consistency**: shape and strides same length
4. **Offset Bounds**: offset < |𝒫|
5. **Refcount Positivity**: refcount(data) ≥ 1 for live views
6. **COW Correctness**: Materialization on shared tensor creates exclusive buffer

---

### 5. Pseudocode Contract Summary

```
Contract(Op, Precond, InputState) → (OutputState, OwnershipEffect, MemoryEffect, FailureReasons)

OP-1: ALLOC(shape)
  Pre: shape non-empty, ∏ shape > 0
  In: N/A
  Out: contiguous=1, own=1, offset=0, refcount=1
  Own: new Arc, refcount=1
  Mem: allocate ∏ shape elements
  Fail: EmptyTensor, OutOfMemory

OP-2: FREE
  Pre: valid
  In: any
  Out: deallocated (refcount--)
  Own: Arc::drop
  Mem: free if refcount→0
  Fail: none

OP-3: VIEW(ranges)
  Pre: ranges valid for shape
  In: any
  Out: own=0, recomputed_contiguity, offset/stride scaled
  Own: refcount++
  Mem: zero-copy
  Fail: DimensionMismatch, InvalidSliceRange

OP-4: TRANSPOSE(perm)
  Pre: perm valid permutation
  In: any
  Out: shape/strides permuted, recomputed_contiguity
  Own: none
  Mem: metadata only
  Fail: InvalidPermutation

OP-5: RESHAPE(new_shape)
  Pre: size preserved, contiguous & exclusive only
  In: contiguous=1, own=1
  Out: new shape, canonical strides, own=1
  Own: none
  Mem: no allocation
  Fail: ShapeMismatch, NotContiguous, NotExclusive

OP-6: MATERIALIZE
  Pre: always valid
  In: any
  Out: contiguous=1, own=1, canonical strides, offset=0
  Own: if needed, new Arc (old refcount--)
  Mem: allocate + copy if non-canonical or shared
  Fail: OutOfMemory

OP-7: GET(coord)
  Pre: coord valid
  In: any
  Out: unchanged
  Own: none
  Mem: read single element
  Fail: DimensionMismatch, IndexOutOfBounds

OP-8: SET(coord, value)
  Pre: coord valid
  In: any
  Out: element written, own updated if COW
  Own: refcount-- (old), refcount++ (new) if COW
  Mem: if refcount>1, copy-on-write; else in-place
  Fail: DimensionMismatch, IndexOutOfBounds
```

---

### 6. References

- Memory Model: See MEMORY_MODEL.md for buffer layout formulas
- Governance State Machine: See GOVERNANCE_STATE_MACHINE.md for state transitions
- Temporal Safety: See TEMPORAL_SAFETY.md for lifetime proofs
- Proof Obligations: See PROOF_OBLIGATIONS.md for formal lemmas
