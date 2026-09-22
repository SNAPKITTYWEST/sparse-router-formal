{ NraayTensor Reference Implementation: Main program
  Demonstrates the Free Pascal reference layer for semantic validation.

  Usage:
    - Runs comprehensive test suite
    - Validates conformance with Rust implementation
    - Provides oracle functions for cross-checking

  Exit codes:
    0 - All tests passed
    1 - Some tests failed
    2 - Fatal error
}

program NraayTensorRef;

{$MODE FPC}
{$APPTYPE CONSOLE}

uses
  SysUtils,
  descriptor,
  buffer,
  tensor_array,
  ownership,
  governance,
  materialize,
  tensor,
  validation,
  tests;

procedure PrintBanner;
begin
  writeln('');
  writeln('================================================================');
  writeln('  NraayTensor Reference Implementation (Free Pascal)');
  writeln('  Semantic validation layer for Rust tensor-core library');
  writeln('================================================================');
  writeln('');
end;

procedure PrintTestResults(const stats: TTestStats);
begin
  writeln('');
  writeln('Test Results:');
  writeln('  Total:  ', stats.total_tests);
  writeln('  Passed: ', stats.passed);
  writeln('  Failed: ', stats.failed);
  writeln('');

  if stats.failed = 0 then
  begin
    writeln('All tests PASSED!');
    halt(0);
  end
  else
  begin
    writeln('Some tests FAILED.');
    halt(1);
  end;
end;

procedure DemoBasicOperations;
var
  t: TNraayTensor;
  t2: TNraayTensor;
  shape: array [0..1] of uint32;
  coord: array [0..1] of uint32;
begin
  writeln('');
  writeln('--- Basic Operations Demo ---');
  writeln('');

  { Create 3x4 tensor }
  shape[0] := 3;
  shape[1] := 4;
  writeln('Creating 3x4 tensor...');
  t := CreateTensor(shape);

  if IsValid(t) then
  begin
    writeln('Tensor created successfully');
    writeln('  Rank: ', Rank(t));
    writeln('  Length: ', Length(t));
    writeln('  B_layout (is_contiguous): ', t.descriptor.is_contiguous);
    writeln('  B_own (owns_storage): ', t.descriptor.owns_storage);
    writeln('  Ownership state: ', StateToString(t.ownership.state));
  end;

  { Set element }
  coord[0] := 1;
  coord[1] := 2;
  writeln('');
  writeln('Setting element at [1,2] = 42.0...');
  SetElement(t, coord, 42.0);

  writeln('Reading element at [1,2] = ', GetElement(t, coord):0:1);

  { Clone }
  writeln('');
  writeln('Cloning tensor...');
  t2 := CloneTensor(t);
  writeln('Refcount after clone: ', GetRefCount(t2.data));
  writeln('Original state: ', StateToString(t.ownership.state));
  writeln('Cloned state: ', StateToString(t2.ownership.state));

  { Materialize }
  writeln('');
  writeln('Materializing cloned tensor...');
  Materialize(t2);
  writeln('After materialization:');
  writeln('  B_layout: ', t2.descriptor.is_contiguous);
  writeln('  B_own: ', t2.descriptor.owns_storage);
  writeln('  Ownership state: ', StateToString(t2.ownership.state));
  writeln('  Offset: ', t2.descriptor.offset);

  DestroyTensor(t);
  DestroyTensor(t2);
  writeln('');
end;

procedure PrintGovernanceSummary;
begin
  writeln('');
  writeln('--- Governance Lemmas ---');
  writeln('');
  writeln('Lemma 1 (Shape-Rank):');
  writeln('  rank(t) = len(shape(t))');
  writeln('');
  writeln('Lemma 2 (Length Product):');
  writeln('  len(t) = product of all shape dimensions');
  writeln('');
  writeln('Lemma 3 (Canonical Strides):');
  writeln('  B_layout=1 => strides are row-major canonical');
  writeln('');
  writeln('Lemma 4 (Exclusive Ownership):');
  writeln('  B_own=1 => Arc refcount=1');
  writeln('');
  writeln('Lemma 5 (Generation Safety):');
  writeln('  descriptor.generation = buffer.generation');
  writeln('');
end;

procedure PrintSemantics;
begin
  writeln('');
  writeln('--- Semantic Layer ---');
  writeln('');
  writeln('Physical Storage (Arc semantics):');
  writeln('  - TensorArray wraps TBuffer with refcount');
  writeln('  - Clone increments refcount without copying data');
  writeln('  - Drop decrements refcount, frees at 0');
  writeln('');
  writeln('Semantic View (Descriptor):');
  writeln('  - Shape: logical dimensions s_0, s_1, ...');
  writeln('  - Strides: step sizes sigma_0, sigma_1, ...');
  writeln('  - Offset: starting position in physical buffer');
  writeln('  - Flags: B_layout (canonical?), B_own (exclusive?)');
  writeln('');
  writeln('Operations (matched with Rust implementation):');
  writeln('  - ALLOC: CreateTensor');
  writeln('  - FREE: DestroyTensor');
  writeln('  - VIEW: Slice (zero-copy via strides)');
  writeln('  - TRANSPOSE: axis permutation');
  writeln('  - MATERIALIZE: COW with contiguity restoration');
  writeln('');
end;

begin
  PrintBanner;

  PrintSemantics;
  PrintGovernanceSummary;

  DemoBasicOperations;

  writeln('');
  writeln('Running comprehensive test suite...');
  PrintTestResults(RunAllTests);
end.
