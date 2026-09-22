# GOVERNANCE_STATE_MACHINE.md
## State Transitions and Invariants for Binary Semantic Governance

### 1. Overview

The governance state machine tracks two binary flags across tensor lifetime:
- **B_layout**: Is the stride pattern canonical (contiguous)?
- **B_own**: Does this view exclusively own the buffer (refcount = 1)?

Together they define 4 permissible states. This document specifies:
- State diagram with transitions
- Pre/post-conditions for every transition
- Invariant conditions (must hold for all states)
- Generation counter management
- Canonical form definition

---

### 2. State Space Definition

**State = (B_layout, B_own)**

```
State 00: SHARED_NON_CANONICAL
         (B_layout=0, B_own=0)
         Logical: Multiple views with non-contiguous access pattern

State 01: SHARED_CANONICAL
         (B_layout=1, B_own=0)
         Logical: Multiple views with contiguous logical access (rare)

State 10: EXCLUSIVE_NON_CANONICAL
         (B_layout=0, B_own=1)
         Logical: Single owner, but transposed/reshaped

State 11: EXCLUSIVE_CANONICAL
         (B_layout=1, B_own=1)
         Logical: Single owner, fully contiguous (creation state)
```

---

### 3. State Diagram

```
                      Creation (ALLOC)
                            |
                            v
                    ┌──────────────┐
                    │ 11: Exclusive│
                    │  Canonical   │
                    │ B=1, B=1     │
                    └──────┬───────┘
                           |
                 ┌─────────┼─────────┐
                 |         |         |
            VIEW()   TRANSPOSE()  MATERIALIZE()
                 |         |         |
                 v         v         v
            ┌────────┐ ┌────────┐ ┌────────┐
        SET/DROP->  | DROP    |
            │        │        │
            v        v        v
         00/01    10      11
```

**Detailed Transitions**:

#### Transition 11 → 00 (via VIEW)

```
Pre:  B_layout=1, B_own=1, valid slice ranges

Action:
  - Arc::clone (refcount++)
  - Compute new offset, strides
  - Recompute B_layout for new shape
  - Set B_own = 0 (always)

Post: (B_layout', 0) where B_layout' ∈ {0, 1}
  - If new stride pattern canonical: → 01
  - If new stride pattern non-canonical: → 00

Semantics:
  - Creates shared view into parent buffer
  - Refcount management ensures parent remains valid
```

#### Transition 11 → 10 (via TRANSPOSE)

```
Pre:  B_layout=1, B_own=1
      perm is valid permutation

Action:
  - Permute shape and strides
  - Recompute B_layout
  - B_own unchanged (still exclusive)

Post: (B_layout', 1) where B_layout' typically = 0
  - If perm is identity: → 11
  - Otherwise: → 10

Semantics:
  - Pure logical permutation
  - Exclusive ownership preserved
  - Physical buffer untouched
```

#### Transition (*, 0) → 11 (via MATERIALIZE)

```
Pre:  B_layout ∈ {0, 1}, B_own = 0
      OR B_layout = 0, B_own = 1

Action:
  - Allocate new Arc<Vec>
  - Copy viewed elements to new buffer
  - Set strides to canonical
  - Set offset = 0
  - Release old Arc (refcount--)
  - Create new Arc (refcount = 1)

Post: B_layout=1, B_own=1

Semantics:
  - Copy-on-write semantics
  - Isolation: other views sharing old buffer unaffected
  - COW barrier prevents memory interference
  - All materialized tensors are contiguous & exclusive
```

#### Transition 10 → 11 (MATERIALIZE optimization)

```
Pre:  B_layout=0, B_own=1

Action:
  - Same as above, but old refcount = 1
  - Deallocate old buffer (was exclusive, now replaced)

Post: B_layout=1, B_own=1

Semantics:
  - No refcount-related interference
  - Exclusive, non-contiguous tensor becomes contiguous
```

#### Transition 01 → 11 (MATERIALIZE no-op)

```
Pre:  B_layout=1, B_own=0

Action:
  - MATERIALIZE returns immediately (idempotent)
  - No allocation, copy, or Arc changes

Post: B_layout=1, B_own=0 (unchanged)

Semantics:
  - Canonical shared view requires no materialization
  - Remains shared (other views unaffected)
```

#### Transition 11 → 11 (MATERIALIZE no-op)

```
Pre:  B_layout=1, B_own=1

Action:
  - MATERIALIZE returns immediately (idempotent)

Post: B_layout=1, B_own=1 (unchanged)

Semantics:
  - Already in desired state
  - No work needed
```

#### Transition (any) → (own'=0) (via DROP/VIEW)

```
Pre:  B_layout ∈ {0, 1}, B_own ∈ {0, 1}

Action (VIEW):
  - Arc::clone
  - refcount++
  - B_own' = 0

Action (DROP):
  - Arc::drop
  - refcount--
  - If refcount → 0: deallocate buffer

Post: B_own' = 0 (if VIEW)
      or deallocate if refcount→0 (if DROP)

Semantics:
  - Ownership refcount management
  - Multiple views from single source
  - Deallocation only when last holder drops
```

---

### 4. Invariant Conditions (Must Hold In All States)

**Inv-1: Governance Correctness**
```
B_layout = 1  ⟺  ∀ i: strides[i] = ∏_{j>i} shape[j]
B_own = 1     ⟺  Arc::strong_count(data) = 1
```

**Inv-2: Shape-Stride Dimension Consistency**
```
shape.len() = strides.len() ∧ shape.len() > 0
```

**Inv-3: Offset Bounds**
```
offset < data.len() ∨ (len() = 0)
```

**Inv-4: Refcount Positivity**
```
Arc::strong_count(data) ≥ 1  ∀ live view
```

**Inv-5: View Containment**
```
∀ logical coord ∈ shape: physical_addr(coord) < data.len()
where physical_addr(x₀, ..., x_{d-1}) = offset + Σ xᵢ * strideᵢ
```

**Inv-6: COW Isolation**
```
If view A materializes while shared (refcount > 1):
  ∀ other view B sharing original buffer:
    A's buffer ≠ B's buffer  (different Arc pointers)
    B still reads original data (old Arc valid)
```

**Inv-7: Temporal Safety**
```
View lifetime ⊆ buffer lifetime via Arc strong_count > 0
```

---

### 5. Generation Counter Management

**Purpose**: Detect stale descriptor references after materialization.

**Generation Counter (u64)**:
- Incremented on every materialization or exclusive write
- Provides fast staleness check without full Arc recount

**Mechanics**:

```rust
struct NraayTensor {
    data: Arc<Vec<f32>>,
    generation: u64,  // Incremented on modification
    // ... other fields
}
```

**Transition Rules**:

```
ALLOC:           generation = 0
VIEW:            generation' = generation (inherited from parent)
TRANSPOSE:       generation unchanged (metadata only)
MATERIALIZE:     generation++ (creates new generation)
SET (if COW):    generation++ (creates new generation)
SET (exclusive): generation unchanged (in-place mutation)
GET:             generation unchanged (read-only)
```

**Staleness Check**:

```
Before reading/writing, consumer validates:
  if (view.generation < parent.generation):
    return Err("Stale view: parent materialized since fork")
```

**Benefit**:
- O(1) check to detect broken generational invariants
- Prevents use-after-materialization bugs
- Orthogonal to Arc-based lifetime safety

---

### 6. Binary Canonical Form

**Definition**: A tensor is in canonical form if B_layout = 1.

**Canonical Strides Formula**:
```
For shape (s₀, s₁, ..., s_{d-1}):
  σᵢ = ∏_{j>i} sⱼ
```

**Examples**:

```
shape = [2, 3]
canonical_strides = [3, 1]

shape = [2, 3, 4]
canonical_strides = [12, 4, 1]

shape = [10]
canonical_strides = [1]

shape = []
canonical_strides = []
```

**Recomputation Algorithm**:

```
fn recompute_canonical(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }
    strides
}

fn is_canonical(shape: &[usize], strides: &[usize]) -> bool {
    recompute_canonical(shape) == strides
}
```

---

### 7. State Reachability Matrix

Which transitions are legal from each state?

```
From / To     00  01  10  11
      00      N   N   N   ✓ (MATERIALIZE)
      01      N   N   N   ✓ (MATERIALIZE) or ✓ (no-op)
      10      N   N   N   ✓ (MATERIALIZE)
      11      ✓   ✓   ✓   ✓ (no-op or already there)
      
✓ = reachable
N = not reachable (cannot happen via contract)
```

**Proof of Safety**:
- ALLOC starts in 11
- VIEW can only exit 11 (refcount > 1 ⟹ B_own = 0)
- TRANSPOSE can only exit 11 if perm ≠ identity (B_layout = 0)
- MATERIALIZE can reach 11 from any state
- No direct transition from 00/01/10 back to 00/01/10 without MATERIALIZE
- Therefore, only reachable states are: {00, 01, 10, 11} with valid paths

---

### 8. Pseudocode State Machine

```
State Machine(tensor: NraayTensor, op: Operation):

  current_state = (B_layout, B_own)

  case op of:
    ALLOC(shape):
      assert(preconditions)
      tensor.shape = shape
      tensor.strides = canonical_strides(shape)
      tensor.offset = 0
      tensor.data = Arc::new(vec![0.0; ∏shape])
      tensor.B_layout = true
      tensor.B_own = true
      tensor.generation = 0
      return (11, invariants_hold)

    VIEW(ranges):
      assert(preconditions)
      view.data = Arc::clone(&tensor.data)
      view.shape = [ranges[i].end - ranges[i].start for i in 0..d]
      view.strides = [tensor.strides[i] * ranges[i].step for i in 0..d]
      view.offset = tensor.offset + Σ (ranges[i].start * tensor.strides[i])
      view.B_layout = is_canonical(view.shape, view.strides)
      view.B_own = false
      view.generation = tensor.generation
      return ((view.B_layout, 0), invariants_hold)

    TRANSPOSE(perm):
      assert(preconditions)
      tensor.shape = [tensor.shape[i] for i in perm]
      tensor.strides = [tensor.strides[i] for i in perm]
      tensor.B_layout = is_canonical(tensor.shape, tensor.strides)
      // B_own unchanged, offset unchanged
      return ((tensor.B_layout, B_own), invariants_hold)

    MATERIALIZE:
      assert(preconditions: always valid)
      
      if (B_layout == 1 && B_own == 1):
        return (11, no_op)
      
      old_data = tensor.data
      new_buffer = vec![0.0; len(tensor)]
      copy_viewed_to(tensor, new_buffer)
      
      tensor.data = Arc::new(new_buffer)
      tensor.strides = canonical_strides(tensor.shape)
      tensor.offset = 0
      tensor.B_layout = true
      tensor.B_own = true
      tensor.generation += 1
      
      // Old data refcount-- (Arc::drop on old_data)
      
      return (11, invariants_hold)

    SET(coord, value):
      assert(preconditions)
      
      if (Arc::strong_count(tensor.data) > 1):
        // COW path
        old_data = tensor.data
        tensor.data = Arc::new((*old_data).clone())
        tensor.B_own = true
        tensor.generation += 1
      
      idx = tensor.offset + Σ (coord[i] * strides[i])
      Arc::get_mut(&mut tensor.data)[idx] = value
      
      return ((B_layout, 1), invariants_hold)

    GET(coord):
      assert(preconditions)
      idx = tensor.offset + Σ (coord[i] * strides[i])
      return (data[idx], state_unchanged)

    DROP:
      Arc::drop(&mut tensor.data)
      if (Arc::strong_count(...) → 0):
        deallocate buffer
      return (deallocated or refcount--)
```

---

### 9. Failure Modes (Prevent Invalid Transitions)

**Invalid Transition Attempts**:

```
RESHAPE on B_layout ≠ 1:
  Error: "Cannot reshape non-contiguous tensor"
  ⟹ Prevent: 00,01,10 → RESHAPE
  ⟹ Only 11 → RESHAPE

VIEW with invalid ranges:
  Error: "Slice range out of bounds"
  ⟹ Prevent invalid (start, end, step)

TRANSPOSE with invalid perm:
  Error: "Invalid permutation"
  ⟹ Prevent non-permutation perm

Stale Generation:
  Error: "View generated before parent materialization"
  ⟹ Prevent use of view with generation < current_generation
```

---

### 10. References

- Tensor Contract: TENSOR_CONTRACT.md (operation specifications)
- Memory Model: MEMORY_MODEL.md (physical layout)
- Temporal Safety: TEMPORAL_SAFETY.md (lifetime proofs)
- Proof Obligations: PROOF_OBLIGATIONS.md (formal lemmas PO-1 to PO-8)
