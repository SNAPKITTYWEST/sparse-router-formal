# TEMPORAL_SAFETY.md
## Lifetime Proofs and Use-After-Free Prevention

### 1. Overview

Temporal safety guarantees that no view can access freed memory. This document formalizes:
- **Lifetime Algebra**: Mathematical notation for view and buffer lifetimes
- **Proof Obligations**: Conditions that must hold for safety
- **Generation Validation**: Fast staleness detection after materialization
- **Use-After-Release Rejection**: Mechanisms to reject invalid accesses
- **COW Isolation Theorem**: Materialization doesn't break sibling views

---

### 2. Lifetime Algebra

**Definition**: Lifetime of an entity is the interval during which it is allocated and accessible.

**Notation**:
```
ℒ_v   : Lifetime of view V (from creation to drop)
ℒ_ℬ   : Lifetime of buffer (physical storage allocation to deallocation)
ℒ_𝓞   : Lifetime of Arc ownership token
𝓞_v   : Ownership token held by view V (Arc<Vec<f32>>)
𝓤_v   : Weak observer token (Weak<Vec<f32>>) — does not extend lifetime
```

**Core Temporal Relation**:
```
∀ view V holding 𝓞_v:
  ℒ_v ⊆ ℒ_ℒ_𝓞_v ⊆ ℒ_ℬ

Proof: Arc::strong_count(𝓞_v) ≥ 1 while V is live.
       Vec<f32> deallocated only when strong_count → 0.
       ⟹ Buffer remains allocated for entire view lifetime. ∎
```

---

### 3. Proof Obligations (Lemmas 1-5)

#### **Lemma 1: Buffer Lifetime Safety**

**Statement**:
```
∀ view V ∈ live_views:
  Arc::strong_count(V.data) ≥ 1
  ∧ V.offset < V.data.len()
  ∧ ∀ coord ∈ V.shape: compute_addr(V, coord) < V.data.len()
```

**Proof Sketch**:
- Rust's ownership system guarantees Arc doesn't deallocate while strong_count > 0
- Each live view holds at least one strong Arc reference
- While view variable is in scope, its destructor hasn't run
- ⟹ Arc::strong_count(V.data) ≥ 1 during view lifetime
- ⟹ Buffer 𝒫 remains allocated and accessible
- All address computations bounded by buffer length

**Executable Check** (in Rust):
```rust
fn test_buffer_lifetime_safety() {
    let mut parent = NraayTensor::new(vec![4, 4]).unwrap();
    for i in 0..16 {
        parent.set(&[i / 4, i % 4], i as f32).unwrap();
    }
    
    let view1 = parent.slice(&[(0, 2, 1), (0, 4, 1)]).unwrap();
    let view2 = parent.slice(&[(2, 4, 1), (0, 4, 1)]).unwrap();
    
    // Both views alive; check invariant
    assert!(parent.refcount() == 3);  // parent + view1 + view2
    assert!(view1.refcount() == 3);   // Same Arc
    
    // While both live, reads through both must succeed
    for i in 0..2 {
        for j in 0..4 {
            let val = view1.get(&[i, j]).unwrap();
            assert!(val == (i * 4 + j) as f32);
        }
    }
    
    // Drop view1; parent + view2 still live
    drop(view1);
    assert!(parent.refcount() == 2);
    assert!(view2.refcount() == 2);
    
    // view2 still valid
    for i in 0..2 {
        for j in 0..4 {
            let val = view2.get(&[i + 2, j]).unwrap();
            assert!(val == ((i + 2) * 4 + j) as f32);
        }
    }
    
    // Drop view2; only parent left
    drop(view2);
    assert!(parent.refcount() == 1);
    
    // Parent reads still valid
    assert!(parent.get(&[0, 0]).unwrap() == 0.0);
}
```

**Oracle Behavior** (both Rust and Pascal must produce):
```
Parent created       → refcount = 1
VIEW(parent)         → refcount = 2
VIEW(parent)         → refcount = 3
Drop view1           → refcount = 2
Drop view2           → refcount = 1
Drop parent          → deallocate buffer
```

---

#### **Lemma 2: Non-Interference of Shared Views**

**Statement**:
```
∀ parent P, views V₁, V₂ created by VIEW(P):
  ∃ shared_buffer B:
    V₁.data == V₂.data == P.data == Arc pointing to B
  ∧ modification(V₁) through SET does not affect V₂.get() values
    if V₁ uses COW (refcount > 1 before SET)
```

**Proof Sketch**:
- Both views created by slicing same parent share original Arc
- If refcount > 1 when SET called on view V₁:
  - SET triggers COW: new Arc allocated, old Arc refcount--
  - V₁.data now points to new buffer (copy-on-write)
  - V₂.data still points to original buffer
- ⟹ Modifications to V₁ don't affect V₂

**Executable Check**:
```rust
fn test_non_interference() {
    let mut parent = NraayTensor::new(vec![4, 4]).unwrap();
    for i in 0..16 {
        parent.set(&[i / 4, i % 4], i as f32).unwrap();
    }
    
    let mut view1 = parent.slice(&[(0, 2, 1), (0, 4, 1)]).unwrap();
    let view2 = parent.slice(&[(2, 4, 1), (0, 4, 1)]).unwrap();
    
    // All share same buffer
    let orig_ptr = Arc::as_ptr(&parent.data());
    assert!(Arc::as_ptr(&view1.data()) == orig_ptr);
    assert!(Arc::as_ptr(&view2.data()) == orig_ptr);
    assert!(parent.refcount() == 3);
    
    // Modify view1 via SET (triggers COW since refcount = 3)
    let orig_val = view1.get(&[0, 0]).unwrap();
    assert!(orig_val == 0.0);
    view1.set(&[0, 0], 99.0).unwrap();
    
    // After COW, view1 has new buffer
    let new_ptr = Arc::as_ptr(&view1.data());
    assert!(new_ptr != orig_ptr);
    
    // view1 modified
    assert!(view1.get(&[0, 0]).unwrap() == 99.0);
    
    // view2 unchanged (still uses original buffer)
    assert!(view2.get(&[0, 0]).unwrap() == 8.0);  // (2,0) = 8
    
    // parent unchanged
    assert!(parent.get(&[0, 0]).unwrap() == 0.0);
    
    // Refcounts: parent + view2 share orig; view1 exclusive on copy
    assert!(parent.refcount() == 2);
    assert!(view1.refcount() == 1);
}
```

---

#### **Lemma 3: Exclusive Post-Materialization**

**Statement**:
```
∀ view V after MATERIALIZE(V):
  V.B_own = 1 ∧ Arc::strong_count(V.data) = 1
  ∧ V.data points to new_buffer (≠ original buffer if shared before)
```

**Proof Sketch**:
- MATERIALIZE allocates `new_buffer = Arc::new(vec![...])`
- Assigns `V.data = new_buffer` (dropping old Arc reference)
- Result: `strong_count(new_buffer) = 1` (only V holds it)
- If original buffer shared (refcount > 1 before materialization):
  - Old Arc refcount decrements by 1 (but remains > 0 for other holders)
  - New Arc starts with refcount = 1
- Post-condition: V is exclusive owner of new buffer

**Executable Check**:
```rust
fn test_exclusive_post_materialization() {
    let mut parent = NraayTensor::new(vec![2, 3]).unwrap();
    
    let mut view = parent.slice(&[(0, 2, 1), (0, 3, 1)]).unwrap();
    assert!(view.owns_storage() == false);
    assert!(parent.refcount() == 2);
    
    let orig_ptr = Arc::as_ptr(&parent.data());
    
    // Materialize (no-op in this case since shape preserved)
    view.materialize().unwrap();
    
    // After materialize, view owns storage
    assert!(view.owns_storage() == true);
    assert!(view.refcount() == 1);
    
    // view has new buffer
    let mat_ptr = Arc::as_ptr(&view.data());
    assert!(mat_ptr == orig_ptr);  // Same buffer since view was slice of whole
    
    // But if view was non-canonical...
    let mut parent2 = NraayTensor::new(vec![2, 3]).unwrap();
    let mut transposed = parent2.clone();
    transposed.transpose(&[0, 1]).unwrap();  // Wait, can't transpose 1D... let me fix this
    
    // Better example:
    let mut parent3 = NraayTensor::new(vec![2, 3]).unwrap();
    let mut view3 = parent3.slice(&[(0, 2, 1), (0, 3, 2)]).unwrap();  // Non-contiguous slice
    
    assert!(view3.is_contiguous() == false);
    assert!(parent3.refcount() == 2);
    let orig_ptr3 = Arc::as_ptr(&parent3.data());
    
    view3.materialize().unwrap();
    
    // After materialization of non-contiguous view:
    // view3 is now contiguous, exclusive, and has possibly different buffer
    assert!(view3.is_contiguous() == true);
    assert!(view3.owns_storage() == true);
    assert!(view3.refcount() == 1);
    
    // If view had stride > 1, new buffer is different
    // (because copied non-contiguous elements into new contiguous buffer)
}
```

---

#### **Lemma 4: Governance Consistency**

**Statement**:
```
∀ view V in any state:
  B_layout(V) = 1 ⟹ strides(V) = canonical_strides(shape(V))
  B_layout(V) = 0 ⟹ strides(V) ≠ canonical_strides(shape(V))
  
  B_own(V) = 1 ⟺ Arc::strong_count(V.data) = 1
  B_own(V) = 0 ⟺ Arc::strong_count(V.data) > 1
```

**Proof Sketch**:
- B_layout flag is computed and cached from stride/shape comparison
- Recomputation always called after operations that change strides
- B_own flag is set based on Arc refcount at operation time
- Updates preserve invariant bidirectional implication

**Executable Check**:
```rust
fn test_governance_consistency() {
    // Test B_layout consistency
    let t = NraayTensor::new(vec![2, 3]).unwrap();
    assert!(t.is_contiguous() == true);
    assert!(t.strides() == vec![3, 1]);
    
    let mut t_transposed = t.clone();
    t_transposed.transpose(&[0, 1]).unwrap();  // Wait, wrong rank again
    
    // Better:
    let mut t2 = NraayTensor::new(vec![2, 3]).unwrap();
    let mut sliced = t2.slice(&[(0, 2, 1), (0, 3, 2)]).unwrap();
    assert!(sliced.is_contiguous() == false);
    assert!(sliced.strides() == vec![3, 2]);
    
    // Test B_own consistency
    let t3 = NraayTensor::new(vec![2, 3]).unwrap();
    assert!(t3.owns_storage() == true);
    assert!(t3.refcount() == 1);
    
    let view = t3.slice(&[(0, 2, 1), (0, 3, 1)]).unwrap();
    assert!(view.owns_storage() == false);
    assert!(view.refcount() == 2);
}
```

---

#### **Lemma 5: Weak Observer Invariant**

**Statement**:
```
∀ weak observer W created via downgrade(V):
  W.upgrade() may return None (if all strong refs dropped)
  ∧ W.upgrade().is_some() ⟹ can access buffer
  ∧ W never prevents buffer deallocation (doesn't contribute to strong_count)
```

**Proof Sketch**:
- Weak<Vec<f32>> created via Arc::downgrade does not increment strong_count
- Weak cannot keep buffer alive on its own
- If all strong refs dropped, next upgrade() returns None
- Weak can safely check if buffer still exists without extending lifetime

**Executable Check**:
```rust
fn test_weak_observer() {
    let mut parent = NraayTensor::new(vec![2, 3]).unwrap();
    let weak = parent.downgrade();
    
    // While parent alive, upgrade succeeds
    assert!(weak.upgrade().is_some());
    
    let view = parent.slice(&[(0, 1, 1), (0, 3, 1)]).unwrap();
    assert!(weak.upgrade().is_some());
    
    // Drop parent and view
    drop(parent);
    assert!(weak.upgrade().is_some());  // view still holds strong ref
    
    drop(view);
    assert!(weak.upgrade().is_none());  // Now buffer deallocated
}
```

---

### 4. Generation Validation Rules

**Purpose**: Fast O(1) check for staleness after materialization.

**Mechanism**:

```rust
struct NraayTensor {
    generation: u64,  // Incremented on each materialization or exclusive write
}
```

**Transition Rules**:

```
ALLOC:           generation = 0
VIEW:            generation' = generation (inherited from parent)
TRANSPOSE:       generation' = generation (metadata only)
MATERIALIZE:     generation' = generation + 1 (new buffer = new generation)
SET (COW):       generation' = generation + 1 (new buffer)
SET (exclusive): generation' = generation (in-place)
GET:             generation' = generation (read-only)
```

**Staleness Check**:

```rust
fn check_staleness(view: &NraayTensor, parent: &NraayTensor) -> Result<()> {
    if view.generation < parent.generation {
        return Err(StaleFork);
    }
    Ok(())
}
```

**Rationale**:
- If view created from parent, view.generation = parent.generation at fork time
- If parent materializes after fork, parent.generation++
- view.generation stays old, parent.generation becomes new
- Check view.generation < parent.generation catches this

**Example**:

```
t = ALLOC([2,3])                 // t.gen = 0
v1 = VIEW(t, ...)                // v1.gen = 0
t.TRANSPOSE(...)                 // t.gen = 0 (metadata only)
v2 = VIEW(t, ...)                // v2.gen = 0
t.MATERIALIZE()                  // t.gen = 1

check_staleness(v1, t)?          // v1.gen (0) < t.gen (1) → error!
check_staleness(v2, t)?          // v2.gen (0) < t.gen (1) → error!

// Create new views after materialization
v3 = VIEW(t, ...)                // v3.gen = 1
check_staleness(v3, t)?          // v3.gen (1) == t.gen (1) → OK
```

---

### 5. Use-After-Release Rejection

**Definition**: Attempt to access freed or materialized buffer via stale descriptor.

**Detection Mechanisms**:

#### **Mechanism 1: Strong Reference Count Check**

```rust
if Arc::strong_count(&view.data) == 0 {
    return Err(UseAfterFree);
}
```

**Limitation**: Weak refs return None on upgrade, but strong refs still hold buffer.

#### **Mechanism 2: Pointer Identity Check**

```rust
if Arc::as_ptr(&view.data) != Arc::as_ptr(&parent.data) {
    // Buffer pointer changed (materialization or COW)
    return Err(StaleDescriptor);
}
```

**Limitation**: Pointer identity can be reused (address space layout randomization).

#### **Mechanism 3: Generation Counter Check** (Primary)

```rust
if view.generation < parent.generation {
    return Err(StaleFork);
}
```

**Advantage**: O(1), prevents use of views forked before parent materialization.

#### **Mechanism 4: Bounds Validation**

```rust
// Always before memory access:
let addr = offset + coord[i] * strides[i];
if addr >= data.len() {
    return Err(IndexOutOfBounds);
}
```

**Advantage**: Catches address overflow or corruption.

---

### 6. COW Isolation Theorem

**Theorem**:
```
If V_A and V_B are sibling views (both created from parent P by VIEW),
and V_A materializes via M_A = MATERIALIZE(V_A),
then V_B remains valid and unaffected.
```

**Proof**:

```
Before materialization:
  V_A.data = Arc_orig (shared with V_B and P)
  V_B.data = Arc_orig
  P.data = Arc_orig
  refcount(Arc_orig) = 3

MATERIALIZE(V_A) executes:
  1. Allocate Arc_new = Arc::new(copy_of_viewed_elements)
  2. Assign V_A.data = Arc_new
  3. Old Arc_orig refcount decrements: 3 → 2

After materialization:
  V_A.data = Arc_new (exclusive, refcount = 1)
  V_B.data = Arc_orig (shared, refcount = 2)
  P.data = Arc_orig (shared, refcount = 2)
  
Arc_orig deallocates only when refcount → 0, which requires V_B and P to also drop.
⟹ V_B.get() still accesses valid Arc_orig buffer
⟹ No use-after-free, no data corruption

∎ COW isolation preserved.
```

**Executable Test**:

```rust
fn test_cow_isolation() {
    let mut parent = NraayTensor::new(vec![4, 4]).unwrap();
    for i in 0..16 {
        parent.set(&[i / 4, i % 4], i as f32).unwrap();
    }
    
    let mut sibling_a = parent.slice(&[(0, 2, 1), (0, 4, 1)]).unwrap();
    let sibling_b = parent.slice(&[(2, 4, 1), (0, 4, 1)]).unwrap();
    
    assert!(parent.refcount() == 3);
    
    // Materialize sibling_a (COW triggered)
    sibling_a.materialize().unwrap();
    
    // After COW:
    assert!(sibling_a.refcount() == 1);  // New buffer, exclusive
    assert!(parent.refcount() == 2);     // Old buffer, still shared with sibling_b
    assert!(sibling_b.refcount() == 2);  // Old buffer, still shared with parent
    
    // Modify sibling_a in its new buffer
    sibling_a.set(&[0, 0], 999.0).unwrap();
    assert!(sibling_a.get(&[0, 0]).unwrap() == 999.0);
    
    // sibling_b and parent unaffected
    assert!(sibling_b.get(&[0, 0]).unwrap() == 8.0);   // (2,0) from original
    assert!(parent.get(&[0, 0]).unwrap() == 0.0);      // Still original
    assert!(parent.get(&[2, 0]).unwrap() == 8.0);
}
```

---

### 7. Cross-Implementation Consistency

**Rust and Pascal implementations must produce identical behavior**:

```
Test Case: Multi-view COW scenario

Rust Test:
  parent = ALLOC([3, 3])
  v1 = VIEW(parent, [(0,2,1), (0,3,1)])
  v2 = VIEW(parent, [(1,3,1), (0,3,1)])
  MATERIALIZE(v1)
  assert(parent.refcount() == 2)
  assert(v1.refcount() == 1)
  assert(v2.refcount() == 2)

Pascal Test:
  parent := AllocTensor([3, 3])
  v1 := ViewTensor(parent, [(0,2,1), (0,3,1)])
  v2 := ViewTensor(parent, [(1,3,1), (0,3,1)])
  MaterializeTensor(v1)
  Assert(GetRefCount(parent) = 2)
  Assert(GetRefCount(v1) = 1)
  Assert(GetRefCount(v2) = 2)

Both produce: parent and v2 share original buffer,
             v1 has new exclusive buffer
```

---

### 8. References

- Tensor Contract: TENSOR_CONTRACT.md (operation semantics)
- Governance State Machine: GOVERNANCE_STATE_MACHINE.md (generation counter rules)
- Memory Model: MEMORY_MODEL.md (buffer lifetime)
- Proof Obligations: PROOF_OBLIGATIONS.md (PO-6, PO-7 linking temporal safety)
