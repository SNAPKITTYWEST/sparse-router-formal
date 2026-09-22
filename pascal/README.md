# NraayTensor Pascal Reference Implementation

Semantic reference layer for the Rust tensor-core library, implemented in Free Pascal (FPC). This reference implementation serves as an **oracle** for validating the Rust implementation, ensuring correctness through cross-language verification.

## Architecture

### Semantic Stack (Bottom to Top)

```
    Application (tests.pas, validation.pas)
           ↓
    Tensor Operations (tensor.pas)
           ↓
    Materialization & Governance (materialize.pas, governance.pas)
           ↓
    Ownership State Machine (ownership.pas)
           ↓
    Descriptor & Tensor Array (descriptor.pas, tensor_array.pas)
           ↓
    Buffer Management (buffer.pas)
           ↓
    Physical Storage (single[])
```

## Modules

### 1. buffer.pas (Generation-tracked allocation)
- **TBuffer**: Physical storage with refcount and generation
- `AllocateBuffer()`: Allocate with generation tracking
- `CloneBuffer()`: Increment refcount (Arc::clone)
- `ReleaseBuffer()`: Decrement refcount, free at 0
- `IncrementGeneration()`: Temporal safety tracking

**Key Invariants:**
- Refcount > 0 always while allocated
- Generation monotonically increases
- Once freed (refcount=0), generation becomes invalid

### 2. tensor_array.pas (Arc<Vec> semantics)
- **TTensorArray**: Wraps TBuffer with generation tracking
- `CreateTensorArray()`: Allocate new array
- `CloneTensorArray()`: Share buffer (refcount++)
- `DropTensorArray()`: Release (refcount--)
- `CopyOnWrite()`: Clone data if refcount > 1

**Arc Semantics:**
- Multiple references to same buffer (shared)
- Mutation only allowed if exclusively owned (refcount=1)
- Mutation triggers new generation

### 3. descriptor.pas (C-compatible view metadata)
- **TTensorDescriptor**: Packed record (C ABI compatible)
  - `rank`: dimensionality
  - `shape_ptr`: → shape array
  - `strides_ptr`: → strides array
  - `offset`: logical offset in buffer
  - `len`: total elements
  - `is_contiguous` (B_layout): canonical strides?
  - `owns_storage` (B_own): refcount=1?
  - `generation`: temporal safety token

**Index Mapping:** `f(x_0,...,x_{d-1}) = offset + Σ(x_i * σ_i)`

### 4. ownership.pas (OWNED/SHARED/EXCLUSIVE states)
- **TOwnershipState**: Enumeration of ownership modes
- State transitions:
  - `OWNED` (refcount=1, mutable)
  - `SHARED` (refcount>1, immutable, COW)
  - `EXCLUSIVE` (after materialization, mutable)

**State Machine:**
```
OWNED --[clone]--> SHARED
SHARED --[materialize]--> EXCLUSIVE
EXCLUSIVE --[release]--> OWNED (if refcount becomes 1)
```

### 5. governance.pas (5 executable lemmas)
Proves correctness of tensor invariants:

1. **Lemma 1 (Shape-Rank)**: `rank(t) = len(shape)`
2. **Lemma 2 (Length Product)**: `len(t) = ∏ shape_i`
3. **Lemma 3 (Canonical Strides)**: `B_layout=1 ⟹ σ_i = ∏_{j>i} s_j`
4. **Lemma 4 (Exclusive Ownership)**: `B_own=1 ⟹ refcount=1`
5. **Lemma 5 (Generation Safety)**: `descriptor.generation = buffer.generation`

`ValidateAllLemmas()`: Verify all 5 proofs hold at once.

### 6. materialize.pas (Copy-on-write barrier)
Restores contiguity and exclusive ownership:

**Algorithm:**
1. If already materialized (B_layout=1 AND B_own=1): return
2. Allocate new contiguous buffer
3. Copy viewed elements in row-major order
4. Replace Arc with new exclusive buffer
5. Reset strides to canonical
6. Update generation, set B_layout=1, B_own=1

**Property:** "Preserving order, strict and bound"

### 7. tensor.pas (Main tensor operations)
Core operations mirroring Rust implementation:

- **ALLOC**: `CreateTensor(shape)` → new exclusive tensor
- **FREE**: `DestroyTensor(var t)` → decrement refcount
- **VIEW**: `Slice(t, ranges)` → zero-copy view (shared)
- **TRANSPOSE**: permute axes via stride rearrangement
- **MATERIALIZE**: `Materialize(var t)` → restore contiguity
- **GET/SET**: element access via logical coordinates

### 8. validation.pas (Conformance oracle)
Cross-implementation validation functions:

- `CompareDescriptors()`: Check equivalence
- `ValidateAlloc()`: Verify creation invariants
- `ValidateSlice()`: Check slicing semantics
- `ValidateTranspose()`: Verify axis permutation
- `ValidateMaterialize()`: Check materialization result
- `ValidateClone()`: Verify shared ownership
- `VerifyGovernanceLemmas()`: All 5 lemmas hold?

### 9. tests.pas (Comprehensive test suite)
10 core conformance tests:

1. Allocation
2. Element access
3. Clone (refcount=2)
4. Slicing (non-contiguous)
5. Transpose (shape permuted)
6. Materialization (contiguous+exclusive)
7. Generation tracking
8. Governance lemmas
9. Reference counting
10. Composite workflow

### 10. nraay_tensor_ref.pas (Main program)
Demonstration and test runner:
```
./nraay_tensor_ref
```

Exit codes:
- 0: All tests passed
- 1: Some tests failed
- 2: Fatal error

## Operation Semantics

### Allocation (ALLOC)
```pascal
t := CreateTensor([3, 4]);
{ Result:
  - Exclusive ownership (refcount=1)
  - Canonical strides (row-major)
  - B_layout=1, B_own=1, OWNED state
  - Generation=0 }
```

### Slicing (VIEW)
```pascal
ranges[0] := [1, 4, 1];  // start=1, end=4, step=1
ranges[1] := [1, 4, 1];
sliced := Slice(t, ranges);
{ Result:
  - Shared ownership (refcount→2)
  - Non-canonical strides (B_layout=0)
  - Offset adjusted, shape reduced
  - B_own=0, SHARED state
  - Same buffer, different view }
```

### Materialization (MATERIALIZE)
```pascal
Materialize(sliced);
{ Result:
  - Exclusive ownership (new buffer)
  - Canonical strides (row-major)
  - B_layout=1, B_own=1
  - Offset=0, new generation
  - EXCLUSIVE state }
```

### Transpose
```pascal
perm := [1, 0];  // swap axes
transposed := Transpose(t, perm);
{ Result:
  - Shape permuted: [3,4] → [4,3]
  - Strides permuted accordingly
  - B_layout likely=0 (non-canonical)
  - Shares buffer, different view }
```

## Governance Properties

### Binary Semantics (B_layout, B_own)

**B_layout (is_contiguous):**
- 1: Strides follow row-major canonical pattern
- 0: Strides do NOT follow canonical (from slicing/transpose)

**B_own (owns_storage):**
- 1: Exclusive ownership (refcount=1), can mutate
- 0: Shared ownership (refcount>1), immutable

### Generation Safety
Every buffer has monotonic generation counter:
```
buffer.generation = counter at allocation time
descriptor.generation = counter at view creation time

Use-after-free prevention: stale views detected by generation mismatch
```

## Compilation and Usage

### FPC Compilation
```bash
cd pascal
fpc -O2 nraay_tensor_ref.pas -o nraay_tensor_ref
./nraay_tensor_ref
```

### With gdb
```bash
fpc -g nraay_tensor_ref.pas
gdb ./nraay_tensor_ref
```

### Unit Testing in Other Code
```pascal
uses tensor, validation, tests;

procedure MyTest;
var
  t: TNraayTensor;
  report: TTestReport;
begin
  t := CreateTensor([2, 3]);
  report := ValidateAlloc(t);
  
  if report.passed then
    writeln('Test passed: ', report.details)
  else
    writeln('Test failed: ', report.details);
    
  DestroyTensor(t);
end;
```

## Cross-Language Conformance

This Pascal reference validates Rust implementation by:

1. **Operation Matching**: Each Rust op has Pascal equivalent
2. **State Verification**: Oracle functions check Rust output states
3. **Invariant Checking**: 5 governance lemmas verified independently
4. **Generation Tracking**: Temporal safety without language runtime
5. **Memory Semantics**: Arc behavior replicated with explicit refcount

### Testing Against Rust
1. Run Rust operation
2. Export resulting descriptor/buffer to Pascal format
3. Call validation oracle
4. Compare Oracle result with expected behavior

## Semantic Properties

### Invariants (Always True)
```
1. rank(t) = len(shape(t))
2. len(t) = ∏ shape_i
3. B_layout=1 → canonical strides
4. B_own=1 → refcount=1
5. generation(descriptor) = generation(buffer)
```

### State Safety
```
OWNED ← exclusive mutation OK, B_own=1
SHARED ← no mutation, COW barrier
EXCLUSIVE ← exclusive mutation OK, B_own=1
```

### Generation Transitions
```
- Allocation: generation_0 (global_counter++)
- Clone: inherits buffer.generation (same)
- Materialize: generation_new (global_counter++)
- Stale check: generation_mismatch → error
```

## Performance Notes

**Memory:** ~1200 LOC core, ~800 LOC validation
**No runtime overhead:** Direct memory access via strides
**Materialization:** O(n) copy only on COW violation or explicit call

## License

Triple license: MIT OR Apache-2.0 OR GPL-3.0-or-later

This ensures:
- Commercial use (MIT/Apache-2.0)
- Open source compliance (GPL-3.0)
- AI company restrictions (GPL-3.0 clause 5)

## See Also

- [FORMAL_SPEC.md](../FORMAL_SPEC.md) - Mathematical definitions
- [Rust Implementation](../nraay-tensor/src/tensor.rs) - Primary implementation
- [Lemmas](../nraay-tensor/src/lemmas.rs) - Formal proofs
