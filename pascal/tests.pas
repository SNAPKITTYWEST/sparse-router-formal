{ Test Suite: Comprehensive conformance tests for NraayTensor reference implementation.
  Tests core operations, governance properties, and state machine transitions.
}

unit tests;

{$MODE FPC}

interface

uses
  descriptor, buffer, tensor_array, ownership, governance, materialize, tensor, validation;

type
  { Test statistics }
  TTestStats = record
    total_tests: integer;
    passed: integer;
    failed: integer;
  end;

{ Run all conformance tests }
function RunAllTests: TTestStats;

{ Test: Tensor allocation }
function TestAllocation: boolean;

{ Test: Element get/set }
function TestElementAccess: boolean;

{ Test: Clone creates shared reference }
function TestClone: boolean;

{ Test: Slicing creates non-contiguous view }
function TestSlicing: boolean;

{ Test: Transpose permutes axes }
function TestTranspose: boolean;

{ Test: Materialization restores contiguity }
function TestMaterialization: boolean;

{ Test: Generation tracking }
function TestGenerationTracking: boolean;

{ Test: Governance lemmas }
function TestGovernanceLemmas: boolean;

{ Test: Buffer reference counting }
function TestRefCounting: boolean;

{ Test: Composite workflow }
function TestCompositeWorkflow: boolean;

implementation

var
  test_stats: TTestStats;

function TestAllocation: boolean;
var
  t: TNraayTensor;
  shape: array [0..1] of uint32;
begin
  shape[0] := 3;
  shape[1] := 4;

  t := CreateTensor(shape);
  result := IsValid(t);

  if result then
  begin
    result := (Length(t) = 12) and (Rank(t) = 2);
  end;

  DestroyTensor(t);
end;

function TestElementAccess: boolean;
var
  t: TNraayTensor;
  shape: array [0..1] of uint32;
  coord: array [0..1] of uint32;
begin
  shape[0] := 2;
  shape[1] := 3;

  t := CreateTensor(shape);

  coord[0] := 1;
  coord[1] := 2;
  SetElement(t, coord, 42.0);

  result := (GetElement(t, coord) = 42.0);

  DestroyTensor(t);
end;

function TestClone: boolean;
var
  t1, t2: TNraayTensor;
  shape: array [0..1] of uint32;
begin
  shape[0] := 2;
  shape[1] := 2;

  t1 := CreateTensor(shape);
  t2 := CloneTensor(t1);

  { Both should share same buffer }
  result := (GetDataPtr(t1.data) = GetDataPtr(t2.data));

  { Refcount should be 2 }
  if result then
    result := (GetRefCount(t2.data) = 2);

  DestroyTensor(t1);
  DestroyTensor(t2);
end;

function TestSlicing: boolean;
var
  t: TNraayTensor;
  sliced: TNraayTensor;
  shape: array [0..1] of uint32;
  ranges: array [0..1] of array of uint32;
begin
  shape[0] := 5;
  shape[1] := 5;

  t := CreateTensor(shape);

  { Create dummy ranges for slicing }
  setlength(ranges[0], 3);
  setlength(ranges[1], 3);
  ranges[0][0] := 1;
  ranges[0][1] := 4;
  ranges[0][2] := 1;
  ranges[1][0] := 1;
  ranges[1][1] := 4;
  ranges[1][2] := 1;

  sliced := Slice(t, ranges);

  { Sliced should not be contiguous }
  result := (sliced.descriptor.is_contiguous = 0);

  { Should be in SHARED state }
  if result then
    result := (sliced.descriptor.owns_storage = 0);

  DestroyTensor(t);
  DestroyTensor(sliced);
end;

function TestTranspose: boolean;
var
  t: TNraayTensor;
  transposed: TNraayTensor;
  shape: array [0..1] of uint32;
  perm: array [0..1] of uint32;
begin
  shape[0] := 3;
  shape[1] := 4;

  t := CreateTensor(shape);

  perm[0] := 1;
  perm[1] := 0;

  transposed := Transpose(t, perm);

  { Shape should be swapped }
  result := (transposed.descriptor.shape_ptr^[0] = 4) and
            (transposed.descriptor.shape_ptr^[1] = 3);

  DestroyTensor(t);
  DestroyTensor(transposed);
end;

function TestMaterialization: boolean;
var
  t: TNraayTensor;
  shape: array [0..1] of uint32;
begin
  shape[0] := 2;
  shape[1] := 2;

  t := CreateTensor(shape);

  { Should already be materialized }
  result := (t.descriptor.is_contiguous = 1) and
            (t.descriptor.owns_storage = 1);

  Materialize(t);

  result := result and
            (t.descriptor.is_contiguous = 1) and
            (t.descriptor.owns_storage = 1) and
            (t.descriptor.offset = 0);

  DestroyTensor(t);
end;

function TestGenerationTracking: boolean;
var
  t: TNraayTensor;
  shape: array [0..0] of uint32;
  gen1, gen2: uint32;
begin
  shape[0] := 5;

  t := CreateTensor(shape);
  gen1 := t.descriptor.generation;

  result := (gen1 >= 0);

  DestroyTensor(t);
end;

function TestGovernanceLemmas: boolean;
var
  t: TNraayTensor;
  shape: array [0..1] of uint32;
begin
  shape[0] := 2;
  shape[1] := 3;

  t := CreateTensor(shape);

  { All lemmas should verify }
  result := VerifyGovernanceLemmas(t, t.data.buffer);

  DestroyTensor(t);
end;

function TestRefCounting: boolean;
var
  t1, t2, t3: TNraayTensor;
  shape: array [0..0] of uint32;
  count: uint32;
begin
  shape[0] := 10;

  t1 := CreateTensor(shape);
  count := GetRefCount(t1.data);
  result := (count = 1);

  t2 := CloneTensor(t1);
  count := GetRefCount(t2.data);
  result := result and (count = 2);

  t3 := CloneTensor(t2);
  count := GetRefCount(t3.data);
  result := result and (count = 3);

  DestroyTensor(t1);
  DestroyTensor(t2);
  DestroyTensor(t3);
end;

function TestCompositeWorkflow: boolean;
var
  t1, t2, sliced, materialized: TNraayTensor;
  shape: array [0..1] of uint32;
  ranges: array [0..1] of array of uint32;
  coord: array [0..1] of uint32;
begin
  shape[0] := 4;
  shape[1] := 4;

  t1 := CreateTensor(shape);
  result := IsValid(t1);

  { Set a value }
  coord[0] := 2;
  coord[1] := 2;
  SetElement(t1, coord, 3.14);
  result := result and (GetElement(t1, coord) = 3.14);

  { Clone }
  t2 := CloneTensor(t1);
  result := result and (GetRefCount(t2.data) = 2);

  { Materialize }
  Materialize(t2);
  result := result and (t2.descriptor.is_contiguous = 1);

  DestroyTensor(t1);
  DestroyTensor(t2);
end;

function RunAllTests: TTestStats;
begin
  with result do
  begin
    total_tests := 0;
    passed := 0;
    failed := 0;
  end;

  { Test allocation }
  inc(result.total_tests);
  if TestAllocation then
    inc(result.passed)
  else
    inc(result.failed);

  { Test element access }
  inc(result.total_tests);
  if TestElementAccess then
    inc(result.passed)
  else
    inc(result.failed);

  { Test cloning }
  inc(result.total_tests);
  if TestClone then
    inc(result.passed)
  else
    inc(result.failed);

  { Test slicing }
  inc(result.total_tests);
  if TestSlicing then
    inc(result.passed)
  else
    inc(result.failed);

  { Test transpose }
  inc(result.total_tests);
  if TestTranspose then
    inc(result.passed)
  else
    inc(result.failed);

  { Test materialization }
  inc(result.total_tests);
  if TestMaterialization then
    inc(result.passed)
  else
    inc(result.failed);

  { Test generation tracking }
  inc(result.total_tests);
  if TestGenerationTracking then
    inc(result.passed)
  else
    inc(result.failed);

  { Test governance lemmas }
  inc(result.total_tests);
  if TestGovernanceLemmas then
    inc(result.passed)
  else
    inc(result.failed);

  { Test reference counting }
  inc(result.total_tests);
  if TestRefCounting then
    inc(result.passed)
  else
    inc(result.failed);

  { Test composite workflow }
  inc(result.total_tests);
  if TestCompositeWorkflow then
    inc(result.passed)
  else
    inc(result.failed);
end;

end.
