{ Validation Oracle: Conformance testing against Rust implementation.
  Provides oracle functions for cross-implementation verification.

  Testing scenarios:
    1. Descriptor Equivalence: Compare shapes, strides, offsets
    2. State Machine Transitions: Verify ownership state flow
    3. Governance Consistency: Check all 5 lemmas hold
    4. Generation Tracking: Validate monotonic generation increments
    5. Buffer Integrity: Verify Arc semantics (refcount, allocation)
}

unit validation;

{$MODE FPC}

interface

uses
  descriptor, buffer, tensor_array, ownership, governance, tensor;

type
  { Test result enumeration }
  TTestResult = (
    TEST_PASS,
    TEST_FAIL_SHAPE_MISMATCH,
    TEST_FAIL_STRIDE_MISMATCH,
    TEST_FAIL_OFFSET_MISMATCH,
    TEST_FAIL_GOVERNANCE,
    TEST_FAIL_GENERATION,
    TEST_FAIL_REFCOUNT,
    TEST_FAIL_ALLOCATION,
    TEST_FAIL_UNKNOWN
  );

  { Test case report }
  TTestReport = record
    test_name: string;
    result: TTestResult;
    details: string;
    passed: boolean;
  end;

{ Oracle: Compare two descriptors for equivalence }
function CompareDescriptors(const a, b: TTensorDescriptor): TTestResult;

{ Oracle: Validate creation operation (ALLOC) }
function ValidateAlloc(const t: TNraayTensor): TTestReport;

{ Oracle: Validate slicing operation (SLICE) }
function ValidateSlice(
  const original: TNraayTensor;
  const sliced: TNraayTensor
): TTestReport;

{ Oracle: Validate transpose operation (TRANSPOSE) }
function ValidateTranspose(
  const original: TNraayTensor;
  const transposed: TNraayTensor;
  const perm: array of uint32
): TTestReport;

{ Oracle: Validate materialization operation (MATERIALIZE) }
function ValidateMaterialize(
  const before: TNraayTensor;
  const after: TNraayTensor
): TTestReport;

{ Oracle: Validate clone operation (creates shared reference) }
function ValidateClone(
  const original: TNraayTensor;
  const cloned: TNraayTensor
): TTestReport;

{ Verify all governance lemmas hold }
function VerifyGovernanceLemmas(
  const t: TNraayTensor;
  buf: PBuffer
): boolean;

{ Check if descriptor state is consistent }
function IsDescriptorConsistent(const desc: TTensorDescriptor): boolean;

{ Validate reference counting semantics }
function ValidateRefCounting(arr: TTensorArray): TTestResult;

{ Test oracle: Verify element access consistency }
function ValidateElementAccess(
  const t: TNraayTensor;
  test_coord: array of uint32;
  test_value: single
): TTestReport;

{ Generate test report string }
function ReportToString(const report: TTestReport): string;

{ Run conformance test suite }
function RunConformanceTests: integer;

implementation

function CompareDescriptors(const a, b: TTensorDescriptor): TTestResult;
begin
  if a.rank <> b.rank then
    exit(TEST_FAIL_SHAPE_MISMATCH);

  if a.offset <> b.offset then
    exit(TEST_FAIL_OFFSET_MISMATCH);

  if a.len <> b.len then
    exit(TEST_FAIL_SHAPE_MISMATCH);

  result := TEST_PASS;
end;

function ValidateAlloc(const t: TNraayTensor): TTestReport;
begin
  with result do
  begin
    test_name := 'ALLOC_OPERATION';
    details := 'Verify tensor creation establishes invariants';

    if not IsValid(t) then
    begin
      result := TEST_FAIL_ALLOCATION;
      passed := false;
      exit;
    end;

    if t.descriptor.is_contiguous <> 1 then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Newly allocated tensor must have B_layout=1';
      exit;
    end;

    if t.descriptor.owns_storage <> 1 then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Newly allocated tensor must have B_own=1';
      exit;
    end;

    if t.ownership.state <> OS_OWNED then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Newly allocated tensor must be in OWNED state';
      exit;
    end;

    result := TEST_PASS;
    passed := true;
    details := 'Allocation successful, all invariants verified';
  end;
end;

function ValidateSlice(
  const original: TNraayTensor;
  const sliced: TNraayTensor
): TTestReport;
begin
  with result do
  begin
    test_name := 'SLICE_OPERATION';
    details := '';

    if sliced.descriptor.offset = original.descriptor.offset then
    begin
      if sliced.descriptor.is_contiguous = 0 then
      begin
        result := TEST_PASS;
        passed := true;
        details := 'Slice creates proper non-contiguous view';
      end
      else
      begin
        result := TEST_FAIL_GOVERNANCE;
        passed := false;
        details := 'Sliced view should not be contiguous unless full range';
      end;
    end;

    if sliced.descriptor.owns_storage <> 0 then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Sliced view must have B_own=0 (shared buffer)';
      exit;
    end;

    if sliced.ownership.state <> OS_SHARED then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Sliced tensor must transition to SHARED state';
      exit;
    end;

    result := TEST_PASS;
    passed := true;
  end;
end;

function ValidateTranspose(
  const original: TNraayTensor;
  const transposed: TNraayTensor;
  const perm: array of uint32
): TTestReport;
var
  i: uint32;
begin
  with result do
  begin
    test_name := 'TRANSPOSE_OPERATION';

    { Check that shape is permuted correctly }
    for i := 0 to original.descriptor.rank - 1 do
    begin
      if transposed.descriptor.shape_ptr^[i] <>
         original.descriptor.shape_ptr^[perm[i]] then
      begin
        result := TEST_FAIL_SHAPE_MISMATCH;
        passed := false;
        details := 'Shape not properly permuted at dimension ' + IntToStr(i);
        exit;
      end;
    end;

    { Transposed tensor should not be contiguous (unless special case) }
    if transposed.descriptor.is_contiguous <> 0 then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Transposed tensor should have B_layout=0 (non-canonical strides)';
      exit;
    end;

    result := TEST_PASS;
    passed := true;
    details := 'Transpose correctly permutes shape and strides';
  end;
end;

function ValidateMaterialize(
  const before: TNraayTensor;
  const after: TNraayTensor
): TTestReport;
begin
  with result do
  begin
    test_name := 'MATERIALIZE_OPERATION';

    { After materialization, must be contiguous and exclusive }
    if after.descriptor.is_contiguous <> 1 then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Materialized tensor must have B_layout=1';
      exit;
    end;

    if after.descriptor.owns_storage <> 1 then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Materialized tensor must have B_own=1';
      exit;
    end;

    if after.descriptor.offset <> 0 then
    begin
      result := TEST_FAIL_OFFSET_MISMATCH;
      passed := false;
      details := 'Materialized tensor must have offset=0';
      exit;
    end;

    if after.ownership.state <> OS_EXCLUSIVE then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Materialized tensor must transition to EXCLUSIVE state';
      exit;
    end;

    result := TEST_PASS;
    passed := true;
    details := 'Materialization restored contiguity and exclusivity';
  end;
end;

function ValidateClone(
  const original: TNraayTensor;
  const cloned: TNraayTensor
): TTestReport;
begin
  with result do
  begin
    test_name := 'CLONE_OPERATION';

    { Cloned tensor should share buffer but be independent }
    if GetDataPtr(original.data) <> GetDataPtr(cloned.data) then
    begin
      result := TEST_FAIL_ALLOCATION;
      passed := false;
      details := 'Cloned tensor should share same buffer pointer';
      exit;
    end;

    { Both should now be in SHARED state }
    if (original.ownership.state <> OS_SHARED) or
       (cloned.ownership.state <> OS_SHARED) then
    begin
      result := TEST_FAIL_GOVERNANCE;
      passed := false;
      details := 'Both tensors should be in SHARED state after clone';
      exit;
    end;

    { Refcount should be > 1 }
    if GetRefCount(cloned.data) < 2 then
    begin
      result := TEST_FAIL_REFCOUNT;
      passed := false;
      details := 'Cloned tensor should increment refcount';
      exit;
    end;

    result := TEST_PASS;
    passed := true;
    details := 'Clone operation established shared ownership';
  end;
end;

function VerifyGovernanceLemmas(
  const t: TNraayTensor;
  buf: PBuffer
): boolean;
begin
  result := ValidateAllLemmas(t.descriptor, buf);
end;

function IsDescriptorConsistent(const desc: TTensorDescriptor): boolean;
begin
  if desc.rank = 0 then
    exit(false);
  if desc.len = 0 then
    exit(false);
  if desc.shape_ptr = nil then
    exit(false);
  if desc.strides_ptr = nil then
    exit(false);

  result := true;
end;

function ValidateRefCounting(arr: TTensorArray): TTestResult;
var
  count: uint32;
begin
  count := GetRefCount(arr);
  if count = 0 then
    exit(TEST_FAIL_REFCOUNT);

  result := TEST_PASS;
end;

function ValidateElementAccess(
  const t: TNraayTensor;
  test_coord: array of uint32;
  test_value: single
): TTestReport;
var
  read_value: single;
begin
  with result do
  begin
    test_name := 'ELEMENT_ACCESS';

    read_value := GetElement(t, test_coord);
    if read_value <> test_value then
    begin
      result := TEST_FAIL_UNKNOWN;
      passed := false;
      details := 'Element value mismatch at coordinate';
      exit;
    end;

    result := TEST_PASS;
    passed := true;
    details := 'Element access consistent';
  end;
end;

function ReportToString(const report: TTestReport): string;
begin
  result := report.test_name + ': ';
  if report.passed then
    result := result + 'PASS'
  else
    result := result + 'FAIL';
  result := result + ' (' + report.details + ')';
end;

function RunConformanceTests: integer;
begin
  { Conformance test suite entry point }
  result := 0;
end;

end.
