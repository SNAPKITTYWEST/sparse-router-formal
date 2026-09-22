# NraayTensor Pascal Reference - Integration Guide

## Quick Start

### Build the Reference Implementation

```bash
cd pascal/
make build
./nraay_tensor_ref
```

Expected output:
```
================================================================
  NraayTensor Reference Implementation (Free Pascal)
  Semantic validation layer for Rust tensor-core library
================================================================

--- Semantic Layer ---
...
Test Results:
  Total:  10
  Passed: 10
  Failed: 0

All tests PASSED!
```

## Using the Reference as an Oracle

The Pascal reference implementation can validate Rust tensor operations through conformance testing.

### Pattern 1: Direct Descriptor Comparison

After a Rust tensor operation, export the resulting descriptor and validate:

```rust
// Rust side
let t = NraayTensor::new(vec![3, 4])?;
let descriptor = t.to_c_descriptor();
```

```pascal
// Pascal side
uses validation;

var
  desc: TTensorDescriptor;
  result: TTestResult;
begin
  // Import descriptor from Rust FFI
  result := CompareDescriptors(expected_desc, desc);
  
  if result = TEST_PASS then
    writeln('Descriptor matches!')
  else
    writeln('Descriptor mismatch: ', result);
end;
```

### Pattern 2: Operation Validation

After each operation (ALLOC, SLICE, TRANSPOSE, MATERIALIZE), call oracle:

```pascal
uses validation, tensor;

// After Rust SLICE operation
sliced := Slice(t, ranges);
report := ValidateSlice(t, sliced);

if report.passed then
  writeln('Slice valid: ', report.details)
else
begin
  writeln('Slice failed: ', report.details);
  halt(1);
end;
```

### Pattern 3: Governance Verification

After any complex operation, verify all 5 lemmas:

```pascal
uses governance, tensor;

var
  buf: PBuffer;
  valid: boolean;
begin
  buf := t.data.buffer;
  valid := ValidateAllLemmas(t.descriptor, buf);
  
  if valid then
    writeln('All governance lemmas verified')
  else
    writeln('Governance violation detected!');
end;
```

### Pattern 4: Conformance Test Suite

For regression testing, import and call Pascal tests:

```pascal
uses tests;

var
  stats: TTestStats;
begin
  stats := RunAllTests;
  writeln('Conformance tests: ', stats.passed, '/', stats.total_tests);
  
  if stats.failed > 0 then
    halt(1);
end;
```

## Integration Scenarios

### Scenario 1: Rust Test → Pascal Validation

```
┌─ Rust Test ──────────────────────────────────────┐
│ 1. Create tensor                                   │
│ 2. Perform operation (SLICE/TRANSPOSE/etc)       │
│ 3. Export descriptor via C FFI                    │
│ 4. Call Pascal validator                          │
│ 5. Compare result                                  │
└────────────────────────────────────────────────────┘
         ↓ Export TTensorDescriptor
┌─ Pascal Validator ────────────────────────────────┐
│ ValidateXxx(descriptor, buffer)                   │
│ VerifyGovernanceLemmas(descriptor, buffer)        │
│ → Returns TTestResult or TTestReport              │
└────────────────────────────────────────────────────┘
         ↓ Result
┌─ Assertion ───────────────────────────────────────┐
│ Expected == Actual?                               │
└────────────────────────────────────────────────────┘
```

### Scenario 2: CI/CD Integration

```bash
#!/bin/bash

# Build Rust implementation
cd tensor-core/nraay-tensor
cargo build --tests

# Build Pascal reference
cd ../../pascal
make build

# Run Rust tests with Pascal validator
cd ../tensor-core/nraay-tensor
cargo test --features conformance-check

# Exit code from conformance (invokes Pascal)
exit $?
```

### Scenario 3: FFI Cross-Calling

**Rust exposes C interface:**
```rust
#[repr(C)]
pub struct CTensorDescriptor {
    pub rank: u32,
    pub shape_ptr: *const u32,
    pub strides_ptr: *const u32,
    pub offset: u32,
    pub len: u32,
    pub is_contiguous: u8,
    pub owns_storage: u8,
    pub generation: u32,
}

#[no_mangle]
pub extern "C" fn validate_descriptor(desc: *const CTensorDescriptor) -> i32 {
    // Pascal side will call this
    0 // success
}
```

**Pascal calls Rust validator:**
```pascal
{ In validation.pas }

function ValidateWithRust(const desc: TTensorDescriptor): TTestResult;
begin
  // Call Rust validator via C FFI
  if ValidateRustTensor(@desc) = 0 then
    result := TEST_PASS
  else
    result := TEST_FAIL_UNKNOWN;
end;
```

## File Structure for Integration

```
devflow-finance-twin/
├── tensor-core/
│   ├── nraay-tensor/
│   │   ├── src/
│   │   │   ├── lib.rs
│   │   │   ├── tensor.rs
│   │   │   ├── lemmas.rs
│   │   │   └── error.rs
│   │   └── Cargo.toml
│   └── FORMAL_SPEC.md
│
├── pascal/                    ← Reference Implementation
│   ├── buffer.pas             ← Arc/refcount semantics
│   ├── tensor_array.pas       ← Buffer wrapper
│   ├── descriptor.pas         ← C-compatible metadata
│   ├── ownership.pas          ← State machine
│   ├── governance.pas         ← 5 lemmas
│   ├── materialize.pas        ← COW barrier
│   ├── tensor.pas             ← Main operations
│   ├── validation.pas         ← Oracle functions
│   ├── tests.pas              ← Test suite
│   ├── nraay_tensor_ref.pas   ← Main program
│   ├── Makefile
│   ├── README.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── INTEGRATION_GUIDE.md   ← You are here
```

## API Reference for Integration

### Validation Oracles

```pascal
{ Compare descriptors }
function CompareDescriptors(const a, b: TTensorDescriptor): TTestResult;

{ Validate operations }
function ValidateAlloc(const t: TNraayTensor): TTestReport;
function ValidateSlice(const original, sliced: TNraayTensor): TTestReport;
function ValidateTranspose(const original, transposed: TNraayTensor; 
                          const perm: array of uint32): TTestReport;
function ValidateMaterialize(const before, after: TNraayTensor): TTestReport;
function ValidateClone(const original, cloned: TNraayTensor): TTestReport;

{ Verify governance }
function VerifyGovernanceLemmas(const t: TNraayTensor; 
                               buf: PBuffer): boolean;

{ Run all tests }
function RunAllTests: TTestStats;
```

### Core Tensor API

```pascal
{ Allocation & Deallocation }
function CreateTensor(const shape: array of uint32): TNraayTensor;
procedure DestroyTensor(var t: TNraayTensor);

{ Cloning }
function CloneTensor(const t: TNraayTensor): TNraayTensor;

{ Element Access }
function GetElement(const t: TNraayTensor; 
                   const coord: array of uint32): single;
procedure SetElement(var t: TNraayTensor; 
                    const coord: array of uint32; value: single);

{ Operations }
function Slice(const t: TNraayTensor; 
              const ranges: array of array of uint32): TNraayTensor;
function Transpose(const t: TNraayTensor; 
                  const perm: array of uint32): TNraayTensor;
procedure Materialize(var t: TNraayTensor);

{ Queries }
function IsValid(const t: TNraayTensor): boolean;
function Length(const t: TNraayTensor): uint32;
function Rank(const t: TNraayTensor): uint32;
```

## Testing Checklists

### Pre-Integration Checklist

- [ ] Pascal code compiles without errors
- [ ] All 10 unit tests pass
- [ ] All 5 governance lemmas verified
- [ ] Makefile build succeeds
- [ ] Demo program runs successfully
- [ ] No memory leaks (valgrind/gdb if needed)

### Conformance Test Checklist

- [ ] ALLOC operation matches Rust creation
- [ ] Element get/set consistent
- [ ] SLICE creates proper views
- [ ] TRANSPOSE permutes correctly
- [ ] MATERIALIZE restores contiguity
- [ ] CLONE increments refcount
- [ ] Generation tracking monotonic
- [ ] State transitions valid
- [ ] Descriptors equivalent
- [ ] Reference counting correct

### Regression Checklist

- [ ] All previous tests still pass
- [ ] No descriptor mismatches
- [ ] Governance lemmas hold
- [ ] No generation conflicts
- [ ] Refcount semantics maintained
- [ ] Ownership state machine valid

## Debugging Tips

### Verify descriptor structure
```pascal
procedure DebugDescriptor(const desc: TTensorDescriptor);
begin
  writeln('Rank: ', desc.rank);
  writeln('Len: ', desc.len);
  writeln('Offset: ', desc.offset);
  writeln('B_layout: ', desc.is_contiguous);
  writeln('B_own: ', desc.owns_storage);
  writeln('Generation: ', desc.generation);
end;
```

### Check state machine
```pascal
procedure DebugOwnership(const token: TOwnershipToken);
begin
  writeln('State: ', StateToString(token.state));
  writeln('Generation: ', token.generation);
  writeln('Refcount: ', token.refcount);
end;
```

### Validate lemmas
```pascal
procedure DebugLemmas(const t: TNraayTensor; buf: PBuffer);
var
  proof: TLemmaProof;
begin
  proof := ProveShapeRank(t.descriptor);
  if not IsProofValid(proof) then
    writeln('Lemma 1 failed: ', proof.details);
  
  { ... repeat for all 5 lemmas ... }
end;
```

### Trace memory
```pascal
procedure TraceAllocation;
var
  arr: TTensorArray;
begin
  arr := CreateTensorArray(100);
  writeln('Created: refcount=', GetRefCount(arr));
  writeln('Exclusively owned: ', IsExclusivelyOwned(arr));
  DropTensorArray(arr);
end;
```

## Performance Considerations

### Memory Usage
- Per tensor: ~200 bytes overhead (descriptor + arrays)
- Per buffer: 4 + refcount size + allocation
- No significant overhead beyond Rust equivalent

### Compilation Time
- FPC: ~1-2 seconds (full build)
- Rust: ~30-60 seconds (comparable)

### Runtime Performance
- All operations O(1) or O(n) (matching Rust)
- No interpretation overhead
- Direct pointer arithmetic

### Profiling
```bash
# Generate flamegraph (if perf available)
fpc -g nraay_tensor_ref.pas
perf record -g ./nraay_tensor_ref
perf report
```

## Common Integration Issues

### Issue 1: Descriptor Structure Mismatch
**Symptom:** Field offsets don't match between Rust and Pascal
**Solution:** Use `#[repr(C)]` on Rust side, `{$PACKRECORDS C}` on Pascal
**Verification:** Compare sizes with sizeof()/SizeOf()

### Issue 2: Pointer Lifetime
**Symptom:** Stale pointers after materialization
**Solution:** Re-export descriptor after each mutation
**Verification:** Compare generation before/after

### Issue 3: Refcount Tracking
**Symptom:** Unexpected refcount values
**Solution:** Track all Clone/Drop operations
**Verification:** Use ValidateRefCounting() oracle

### Issue 4: Generation Overflow
**Symptom:** Generation counter wraps or causes issues
**Solution:** Use uint32 (4 billion+ generations sufficient)
**Verification:** Log generation values during operations

## Next Steps

1. **Build and test Pascal reference**
   ```bash
   make test
   ```

2. **Export Rust tensor as C descriptor**
   - Implement to_c_descriptor() on NraayTensor
   - Ensure struct layout matches TTensorDescriptor

3. **Call Pascal validators from Rust tests**
   - Link pascal library via C FFI
   - Invoke oracles in test assertions

4. **Run conformance test suite**
   - Compare Rust results with Pascal expectations
   - Report any discrepancies

5. **Document findings**
   - Record any semantic differences
   - Update specs if needed

## References

- [README.md](README.md) - Architecture guide
- [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md) - Implementation details
- [../FORMAL_SPEC.md](../FORMAL_SPEC.md) - Mathematical semantics
- [../nraay-tensor/src/lemmas.rs](../nraay-tensor/src/lemmas.rs) - Formal proofs
- [../nraay-tensor/src/tensor.rs](../nraay-tensor/src/tensor.rs) - Rust reference

## Support

For issues or questions:
1. Check IMPLEMENTATION_SUMMARY.md for known limitations
2. Review test cases for usage examples
3. Verify governance lemmas hold
4. Check generation tracking consistency

---

**Status:** Production ready for conformance testing
**Last Updated:** 2026-09-22
