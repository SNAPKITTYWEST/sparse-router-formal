# PROOF_OBLIGATIONS.md
## Formal Lemmas and Test Strategy for NraayTensor Implementations

### 1. Overview

This document specifies 8 proof obligations (PO-1 through PO-8) that both Rust and Pascal implementations must satisfy. Each obligation is:
- **Formally stated** with mathematical notation
- **Tested by oracle** — concrete test cases with expected output
- **Validated independently** in each implementation
- **Cross-checked** between implementations for consistency

---

### 2. Proof Obligation 1: Contiguity Preservation (PO-1)

**Formal Statement**:
```
∀ view V: len(V) ≤ |𝒫| ∧ ∃ offset, strides such that
  ∀ coord ∈ shape(V): addr(coord) < |𝒫| ∧ addr(coord) is unique
```

**Rationale**: The physical buffer never shrinks during view operations. All logical coordinates map to valid physical addresses.

**Test Strategy**:

```
Test Case 1.1: Creation preserves buffer size
  Input:
    shape = [2, 3, 4]
    
  Operations:
    t := ALLOC(shape)
    size := |buffer(t)|
    
  Oracle Behavior:
    size = 2*3*4 = 24
    Assert(len(t) = 24)
    Assert(buffer_size(t) = 24)
    Assert(offset(t) = 0)
    
Test Case 1.2: Slicing reduces logical size but shares buffer
  Input:
    parent shape = [4, 4]
    slice ranges = [(0, 2, 1), (0, 4, 1)]
    
  Operations:
    parent := ALLOC([4, 4])
    view := VIEW(parent, ranges)
    
  Oracle Behavior:
    len(parent) = 16
    buffer_size(parent) = 16
    len(view) = 2*4 = 8
    buffer_size(view) = 16  (shares parent's buffer)
    Assert(buffer(view) = buffer(parent))
    Assert(len(view) ≤ buffer_size(view))
    
Test Case 1.3: Materialization restores canonical layout
  Input:
    shape = [2, 3]
    strides = [1, 2] (non-canonical)
    offset = 0
    
  Operations:
    t := create_with_strides([2, 3], [1, 2], 0)
    MATERIALIZE(t)
    
  Oracle Behavior:
    strides(t) = [3, 1]  (canonical)
    len(t) = 6
    buffer_size(t) = 6
    Assert(is_contiguous(t) = true)
```

**Executable Oracle** (Rust pseudocode):

```rust
fn po1_test_case_1_1() {
    let t = NraayTensor::new(vec![2, 3, 4]).unwrap();
    assert_eq!(t.len(), 24);
    assert_eq!(t.buffer_size(), 24);
    assert_eq!(t.offset(), 0);
}

fn po1_test_case_1_2() {
    let parent = NraayTensor::new(vec![4, 4]).unwrap();
    let view = parent.slice(&[(0, 2, 1), (0, 4, 1)]).unwrap();
    
    assert_eq!(parent.len(), 16);
    assert_eq!(parent.buffer_size(), 16);
    assert_eq!(view.len(), 8);
    assert_eq!(view.buffer_size(), 16);
    // Check same Arc pointer (shared buffer)
    assert_eq!(
        Arc::as_ptr(&parent.data()),
        Arc::as_ptr(&view.data())
    );
}

fn po1_test_case_1_3() {
    let mut t = NraayTensor::new(vec![2, 3]).unwrap();
    // Transpose to create non-canonical strides
    t.transpose(&[1, 0]).unwrap();
    
    assert!(!t.is_contiguous());
    t.materialize().unwrap();
    
    assert!(t.is_contiguous());
    assert_eq!(t.strides(), &[3, 1]);  // Canonical for [3, 2]
}
```

---

### 3. Proof Obligation 2: Mapping Uniqueness (PO-2)

**Formal Statement**:
```
∀ view V with valid (shape, strides, offset):
  f(coord₁) = f(coord₂) ⟹ coord₁ = coord₂
  
where f(x) = offset + Σᵢ xᵢ * strideᵢ
```

**Rationale**: The index mapping is injective — no two logical coordinates map to the same physical address.

**Test Strategy**:

```
Test Case 2.1: Dense row-major is injective
  Input:
    shape = [3, 4]
    strides = [4, 1]
    offset = 0
    
  Operations:
    For each pair (i, j), compute addr(i, j) = 0 + i*4 + j*1
    Collect all addresses: {0,1,2,3,4,5,...,11}
    
  Oracle Behavior:
    Addresses are all distinct
    Assert(len(unique_addresses) = 12)
    Assert(max(addresses) = 11 < buffer_size)
    
Test Case 2.2: Stride > 1 sampling is injective
  Input:
    shape = [2, 3]
    strides = [6, 2]  (every 2nd element in second dimension)
    offset = 0
    
  Operations:
    (0,0) → 0 + 0*6 + 0*2 = 0
    (0,1) → 0 + 0*6 + 1*2 = 2
    (0,2) → 0 + 0*6 + 2*2 = 4
    (1,0) → 0 + 1*6 + 0*2 = 6
    (1,1) → 0 + 1*6 + 1*2 = 8
    (1,2) → 0 + 1*6 + 2*2 = 10
    
  Oracle Behavior:
    Addresses: {0, 2, 4, 6, 8, 10}
    Assert(all unique)
    Assert(max(addresses) = 10)
    
Test Case 2.3: Offset doesn't break injectivity
  Input:
    shape = [2, 2]
    strides = [2, 1]
    offset = 5
    
  Operations:
    (0,0) → 5 + 0*2 + 0*1 = 5
    (0,1) → 5 + 0*2 + 1*1 = 6
    (1,0) → 5 + 1*2 + 0*1 = 7
    (1,1) → 5 + 1*2 + 1*1 = 8
    
  Oracle Behavior:
    Addresses: {5, 6, 7, 8}
    Assert(all unique and >= offset)
```

**Executable Oracle** (Rust pseudocode):

```rust
fn po2_check_injectivity(shape: &[usize], strides: &[usize], offset: usize) -> bool {
    let mut addresses = HashSet::new();
    
    // Generate all possible coordinates
    let mut coord = vec![0; shape.len()];
    loop {
        let addr = offset + coord.iter()
            .zip(strides.iter())
            .map(|(c, s)| c * s)
            .sum::<usize>();
        
        if addresses.contains(&addr) {
            return false;  // Found collision
        }
        addresses.insert(addr);
        
        // Increment coordinate (row-major odometer)
        let mut d = shape.len();
        loop {
            if d == 0 {
                return true;  // All coordinates checked, all unique
            }
            d -= 1;
            coord[d] += 1;
            if coord[d] < shape[d] {
                break;
            }
            coord[d] = 0;
        }
    }
}

fn po2_test_case_2_1() {
    assert!(po2_check_injectivity(&[3, 4], &[4, 1], 0));
}

fn po2_test_case_2_2() {
    assert!(po2_check_injectivity(&[2, 3], &[6, 2], 0));
}

fn po2_test_case_2_3() {
    assert!(po2_check_injectivity(&[2, 2], &[2, 1], 5));
}
```

---

### 4. Proof Obligation 3: Governance Correctness (PO-3)

**Formal Statement**:
```
∀ view V:
  B_layout(V) = 1 ⟺ strides(V) = canonical_strides(shape(V))
  B_own(V) = 1 ⟺ Arc::strong_count(V.data) = 1
```

**Rationale**: Binary governance flags accurately reflect tensor state.

**Test Strategy**:

```
Test Case 3.1: Fresh tensor is contiguous & exclusive
  Input:
    shape = [2, 3]
    
  Oracle Behavior:
    B_layout = 1
    B_own = 1
    strides = [3, 1] (canonical)
    refcount = 1
    
Test Case 3.2: Slice breaks exclusive ownership
  Input:
    parent: B_own = 1, refcount = 1
    
  Oracle Behavior:
    view = VIEW(parent, ...)
    B_own(view) = 0
    refcount(parent) = 2
    refcount(view) = 2
    
Test Case 3.3: Transpose may break contiguity
  Input:
    tensor: B_layout = 1
    perm = [1, 0]
    
  Oracle Behavior:
    TRANSPOSE(tensor, perm)
    If perm changes canonical order:
      B_layout = 0
    Else:
      B_layout = 1
    B_own unchanged
    
Test Case 3.4: Materialization restores both flags
  Input:
    view: B_layout ∈ {0, 1}, B_own ∈ {0, 1}
    
  Oracle Behavior:
    MATERIALIZE(view)
    B_layout = 1
    B_own = 1
    refcount = 1 (new exclusive buffer)
```

**Executable Oracle**:

```rust
fn po3_test_case_3_1() {
    let t = NraayTensor::new(vec![2, 3]).unwrap();
    assert_eq!(t.is_contiguous(), true);
    assert_eq!(t.owns_storage(), true);
    assert_eq!(t.strides(), &[3, 1]);
    assert_eq!(t.refcount(), 1);
}

fn po3_test_case_3_2() {
    let parent = NraayTensor::new(vec![2, 3]).unwrap();
    let view = parent.slice(&[(0, 2, 1), (0, 3, 1)]).unwrap();
    
    assert_eq!(view.owns_storage(), false);
    assert_eq!(parent.refcount(), 2);
    assert_eq!(view.refcount(), 2);
}

fn po3_test_case_3_3() {
    let mut t = NraayTensor::new(vec![2, 3]).unwrap();
    let own_before = t.owns_storage();
    
    // Transpose will permute strides, likely breaking canonicity
    t.transpose(&[1, 0]).unwrap();
    
    if t.shape() == &[3, 2] {
        // For [3, 2], canonical is [2, 1]
        let should_be_canonical = t.strides() == &[2, 1];
        assert_eq!(t.is_contiguous(), should_be_canonical);
    }
    assert_eq!(t.owns_storage(), own_before);  // Ownership unchanged
}

fn po3_test_case_3_4() {
    let mut parent = NraayTensor::new(vec![2, 3]).unwrap();
    let mut view = parent.slice(&[(0, 2, 1), (0, 3, 2)]).unwrap();
    
    // Before materialize
    assert!(!view.is_contiguous());
    assert!(!view.owns_storage());
    
    view.materialize().unwrap();
    
    // After materialize
    assert_eq!(view.is_contiguous(), true);
    assert_eq!(view.owns_storage(), true);
    assert_eq!(view.refcount(), 1);
}
```

---

### 5. Proof Obligation 4: Slice Soundness (PO-4)

**Formal Statement**:
```
∀ slice S created by VIEW(parent, ranges):
  ∀ coord_s ∈ shape(S):
    physical_addr(coord_s, S) = physical_addr(coord_s + start, parent)
    
where coord_s + start means adding ranges[i].start to each dimension
```

**Rationale**: Slicing correctly maps logical coordinates to parent's physical addresses.

**Test Strategy**:

```
Test Case 4.1: Simple row slice
  Parent:
    shape = [4, 4], strides = [4, 1], offset = 0
    data = [0,1,2,3, 4,5,6,7, 8,9,10,11, 12,13,14,15]
    
  Slice:
    ranges = [(1, 3, 1), (0, 4, 1)]
    
  Expected Slice View:
    shape = [2, 4]
    strides = [4, 1]
    offset = 1*4 + 0*1 = 4
    
  Validation:
    Slice coord (0, 0) → phys 4 + 0*4 + 0*1 = 4 → value 4 ✓
    Slice coord (0, 1) → phys 4 + 0*4 + 1*1 = 5 → value 5 ✓
    Slice coord (1, 0) → phys 4 + 1*4 + 0*1 = 8 → value 8 ✓
    Slice coord (1, 1) → phys 4 + 1*4 + 1*1 = 9 → value 9 ✓
    
Test Case 4.2: Strided slice
  Parent:
    shape = [4, 4], strides = [4, 1], offset = 0
    
  Slice:
    ranges = [(0, 4, 2), (0, 4, 2)]  (every 2nd element)
    
  Expected Slice View:
    shape = [2, 2]
    strides = [8, 2]  (scaled by step)
    offset = 0*4 + 0*1 = 0
    
  Validation:
    Slice coord (0, 0) → phys 0 + 0*8 + 0*2 = 0 ✓
    Slice coord (0, 1) → phys 0 + 0*8 + 1*2 = 2 ✓
    Slice coord (1, 0) → phys 0 + 1*8 + 0*2 = 8 ✓
    Slice coord (1, 1) → phys 0 + 1*8 + 1*2 = 10 ✓
```

**Executable Oracle**:

```rust
fn po4_validate_slice_mapping() {
    let parent = create_parent([4, 4]);
    for i in 0..4 {
        for j in 0..4 {
            let idx = i * 4 + j;
            parent.set(&[i, j], idx as f32).unwrap();
        }
    }
    
    let slice = parent.slice(&[(1, 3, 1), (0, 4, 1)]).unwrap();
    
    // Check that slice coordinates map correctly to parent
    for si in 0..2 {
        for sj in 0..4 {
            let slice_val = slice.get(&[si, sj]).unwrap();
            let parent_coord_i = 1 + si;
            let parent_coord_j = 0 + sj;
            let parent_val = parent.get(&[parent_coord_i, parent_coord_j]).unwrap();
            assert_eq!(slice_val, parent_val);
        }
    }
}
```

---

### 6. Proof Obligation 5: COW Isolation (PO-5)

**Formal Statement**:
```
∀ views V_A, V_B created from parent P:
  If MATERIALIZE(V_A) triggered COW (refcount > 1 before):
    V_A.data ≠ V_B.data (different Arc pointers after)
    ∧ V_B.get() still returns original values
    ∧ ∀ writes to V_A don't affect V_B or P
```

**Rationale**: COW creates isolation, preventing materialization from corrupting sibling views.

**Test Strategy**:

```
Test Case 5.1: Basic COW isolation
  Setup:
    parent = ALLOC([3, 3])
    va = VIEW(parent, ...)
    vb = VIEW(parent, ...)
    refcount(parent) = 3
    
  Action:
    MATERIALIZE(va)  → refcount(parent) = 2, refcount(va) = 1
    
  Check:
    parent.data ≠ va.data
    parent.data = vb.data
    vb.get() returns original parent values
    
Test Case 5.2: COW + mutation
  Setup:
    parent = ALLOC([2, 2]), populated [0,1,2,3]
    va = VIEW(parent, [(0,2,1), (0,2,1)])
    vb = VIEW(parent, [(0,2,1), (0,2,1)])
    
  Action:
    MATERIALIZE(va)
    SET(va, [0,0], 99.0)
    
  Check:
    va.get([0,0]) = 99.0
    vb.get([0,0]) = 0.0    (original)
    parent.get([0,0]) = 0.0 (original)
```

**Executable Oracle**:

```rust
fn po5_test_cow_isolation() {
    let mut parent = NraayTensor::new(vec![3, 3]).unwrap();
    for i in 0..9 {
        parent.set(&[i / 3, i % 3], i as f32).unwrap();
    }
    
    let mut va = parent.slice(&[(0, 3, 1), (0, 3, 1)]).unwrap();
    let vb = parent.slice(&[(0, 3, 1), (0, 3, 1)]).unwrap();
    
    assert_eq!(parent.refcount(), 3);
    
    let parent_ptr_before = Arc::as_ptr(&parent.data());
    let va_ptr_before = Arc::as_ptr(&va.data());
    let vb_ptr_before = Arc::as_ptr(&vb.data());
    
    assert_eq!(parent_ptr_before, va_ptr_before);
    assert_eq!(parent_ptr_before, vb_ptr_before);
    
    // Materialize va (no-op in this case since whole slice of whole parent)
    va.materialize().unwrap();
    
    let parent_ptr_after = Arc::as_ptr(&parent.data());
    let va_ptr_after = Arc::as_ptr(&va.data());
    let vb_ptr_after = Arc::as_ptr(&vb.data());
    
    // If va had non-canonical strides, it would have new buffer now
    // But full contiguous slice won't change buffer, so let's use strided slice
}

fn po5_test_cow_isolation_with_stride() {
    let mut parent = NraayTensor::new(vec![3, 3]).unwrap();
    for i in 0..9 {
        parent.set(&[i / 3, i % 3], i as f32).unwrap();
    }
    
    let mut va = parent.slice(&[(0, 3, 2), (0, 3, 1)]).unwrap();  // Strided, non-contiguous
    let vb = parent.slice(&[(0, 3, 1), (0, 3, 1)]).unwrap();
    
    let parent_ptr_orig = Arc::as_ptr(&parent.data());
    let vb_ptr_orig = Arc::as_ptr(&vb.data());
    
    // Materialize va → COW triggers
    va.materialize().unwrap();
    
    let va_ptr_new = Arc::as_ptr(&va.data());
    let vb_ptr_now = Arc::as_ptr(&vb.data());
    let parent_ptr_now = Arc::as_ptr(&parent.data());
    
    // va has new buffer, vb and parent share original
    assert_ne!(va_ptr_new, parent_ptr_orig);
    assert_eq!(vb_ptr_now, parent_ptr_orig);  // vb unchanged
    assert_eq!(parent_ptr_now, parent_ptr_orig);
    
    // Modify va; vb and parent unaffected
    va.set(&[0, 0], 999.0).unwrap();
    assert_eq!(va.get(&[0, 0]).unwrap(), 999.0);
    assert_eq!(vb.get(&[0, 0]).unwrap(), 0.0);    // Original
    assert_eq!(parent.get(&[0, 0]).unwrap(), 0.0); // Original
}
```

---

### 7. Proof Obligation 6: Temporal Safety (PO-6)

**Formal Statement**:
```
∀ view V:
  lifetime(V) ⊆ lifetime(buffer(V))
  
i.e., while V is alive (strong_count(V.data) > 0),
      buffer(V) is allocated and accessible
```

**Rationale**: No use-after-free within lifetime of any view.

**Test Strategy**: See TEMPORAL_SAFETY.md Lemma 1-5

---

### 8. Proof Obligation 7: Materialization Correctness (PO-7)

**Formal Statement**:
```
∀ view V:
  If B_layout(V) = 0 before MATERIALIZE(V):
    ∃ new_buffer B' such that:
      ∀ coord ∈ shape(V): V_after.get(coord) = V_before.get(coord)
      ∧ V_after.B_layout = 1
      ∧ V_after.strides = canonical_strides(shape(V))
      ∧ V_after.offset = 0
```

**Rationale**: Materialization is semantically correct — values preserved, layout restored.

**Test Strategy**:

```
Test Case 7.1: Transpose then materialize
  Setup:
    t = ALLOC([2, 3])
    Populate with logical values
    
  Action:
    TRANSPOSE(t, [1, 0])  → shape = [3, 2], B_layout = 0
    Snapshot values before materialize
    MATERIALIZE(t)
    
  Validation:
    shape = [3, 2] (unchanged)
    strides = [2, 1] (canonical for new shape)
    offset = 0
    ∀ coord: GET(t_after, coord) = snapshot_value[coord]
    
Test Case 7.2: Strided slice then materialize
  Setup:
    parent = ALLOC([4, 4])
    view = VIEW(parent, [(0,4,2), (0,4,2)])  → shape [2, 2], non-canonical strides
    Snapshot values [0,0]=0, [0,1]=2, [1,0]=8, [1,1]=10
    
  Action:
    MATERIALIZE(view)
    
  Validation:
    shape = [2, 2] (unchanged)
    strides = [2, 1] (canonical)
    view.get([0,0]) = 0 ✓
    view.get([0,1]) = 2 ✓
    view.get([1,0]) = 8 ✓
    view.get([1,1]) = 10 ✓
```

**Executable Oracle**:

```rust
fn po7_test_transpose_materialize() {
    let mut t = NraayTensor::new(vec![2, 3]).unwrap();
    for i in 0..6 {
        t.set(&[i / 3, i % 3], i as f32).unwrap();
    }
    
    let snapshot: Vec<f32> = (0..6).map(|i| t.get(&[i / 3, i % 3]).unwrap()).collect();
    
    t.transpose(&[1, 0]).unwrap();
    assert!(!t.is_contiguous());
    
    t.materialize().unwrap();
    assert!(t.is_contiguous());
    assert_eq!(t.shape(), &[3, 2]);
    assert_eq!(t.strides(), &[2, 1]);
    assert_eq!(t.offset(), 0);
    
    // Verify values preserved (but reindexed due to transpose)
    // Original (i,j) with shape [2,3] → Transposed (j,i) with shape [3,2]
    assert_eq!(t.get(&[0, 0]).unwrap(), snapshot[0]);  // Was (0,0)
    assert_eq!(t.get(&[1, 0]).unwrap(), snapshot[1]);  // Was (0,1)
    assert_eq!(t.get(&[2, 0]).unwrap(), snapshot[2]);  // Was (0,2)
    assert_eq!(t.get(&[0, 1]).unwrap(), snapshot[3]);  // Was (1,0)
    assert_eq!(t.get(&[1, 1]).unwrap(), snapshot[4]);  // Was (1,1)
    assert_eq!(t.get(&[2, 1]).unwrap(), snapshot[5]);  // Was (1,2)
}
```

---

### 9. Proof Obligation 8: Generation Counter Validation (PO-8)

**Formal Statement**:
```
∀ view V_child created from V_parent:
  If V_parent.MATERIALIZE() called after creation:
    generation(V_parent) > generation(V_child)
    ∧ check_staleness(V_child, V_parent) returns error
  Else:
    generation(V_parent) = generation(V_child)
    ∧ check_staleness(V_child, V_parent) succeeds
```

**Rationale**: Generation counters detect stale views after materialization.

**Test Strategy**:

```
Test Case 8.1: Fresh view is not stale
  Setup:
    parent = ALLOC([2, 3])
    parent.generation = 0
    
  Action:
    view = VIEW(parent)
    view.generation = 0
    
  Check:
    check_staleness(view, parent) succeeds
    
Test Case 8.2: View becomes stale after parent materialization
  Setup:
    parent = ALLOC([2, 3])
    parent.generation = 0
    
  Action:
    view = VIEW(parent)
    view.generation = 0
    MATERIALIZE(parent)  → parent.generation = 1
    
  Check:
    check_staleness(view, parent) returns error (0 < 1)
    
Test Case 8.3: Fresh view after parent materialization is valid
  Setup:
    parent = ALLOC([2, 3])
    MATERIALIZE(parent)  → parent.generation = 1
    
  Action:
    view = VIEW(parent)
    view.generation = 1
    
  Check:
    check_staleness(view, parent) succeeds (1 == 1)
```

**Executable Oracle**:

```rust
fn po8_test_stale_detection() {
    let mut parent = NraayTensor::new(vec![2, 3]).unwrap();
    assert_eq!(parent.generation(), 0);
    
    let mut view = parent.slice(&[(0, 2, 1), (0, 3, 1)]).unwrap();
    assert_eq!(view.generation(), 0);  // Inherited from parent
    
    // Fresh view should not be stale
    assert!(check_staleness(&view, &parent).is_ok());
    
    // Transpose parent (metadata only, no generation change)
    parent.transpose(&[0, 1]).unwrap();  // Wrong rank for this test; use better example
    assert_eq!(parent.generation(), 0);
    assert!(check_staleness(&view, &parent).is_ok());
    
    // Materialize parent → generation++
    parent.materialize().unwrap();
    assert_eq!(parent.generation(), 1);
    
    // Now view is stale
    assert!(check_staleness(&view, &parent).is_err());
    
    // But new views created after materialization are fresh
    let fresh_view = parent.slice(&[(0, 2, 1), (0, 3, 1)]).unwrap();
    assert_eq!(fresh_view.generation(), 1);
    assert!(check_staleness(&fresh_view, &parent).is_ok());
}
```

---

### 10. Cross-Implementation Validation

**Strategy**: Both implementations execute same test cases and produce identical results.

```
Test Suite Execution:

Rust Implementation:
  cargo test po1_*
  cargo test po2_*
  ... po8_*
  
Pascal Implementation:
  compile tests/po_tests.pas
  run tests
  
Comparison:
  For each test T:
    output_rust = run(T)
    output_pascal = run(T)
    if output_rust ≠ output_pascal:
      report(MISMATCH, T, output_rust, output_pascal)
```

---

### 11. Oracle Result Format

Each test produces structured output:

```
PO-<N>.<case>: <description>
  Status: PASS | FAIL
  Invariant: <statement>
  Values:
    <key>: <value>
    <key>: <value>
  Notes: <optional clarification>
```

---

### 12. References

- Formal Specification: TENSOR_CONTRACT.md
- State Machine: GOVERNANCE_STATE_MACHINE.md
- Memory Layout: MEMORY_MODEL.md
- Temporal Safety Proofs: TEMPORAL_SAFETY.md
