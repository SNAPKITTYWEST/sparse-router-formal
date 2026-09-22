{ ============================================================================
  NraayTensor FFI ABI Bindings for Free Pascal

  Provides Pascal-idiomatic bindings to the Rust tensor FFI layer.
  All operations are thin wrappers around C FFI calls into tensor_ffi.rs.

  The Pascal implementation remains independent for validation.
  These bindings allow Pascal to invoke Rust tensor operations for
  conformance testing and oracle comparison.
  ============================================================================ }

unit TensorABI;

{$MODE OBJFPC}
{$H+}
{$MODESWITCH ADVANCEDRECORDS}

interface

uses
  Classes, SysUtils, TypInfo;

type

  { ========================================================================
    Tensor Descriptor — mirrors C ABI
    ======================================================================== }

  TTensorDescriptorC = packed record
    buffer_id: UInt64;
    base_addr: UInt64;
    offset: UInt64;
    rank: UInt32;
    element_count: UInt64;
    shape_ptr: UInt64;
    stride_ptr: UInt64;
    governance: Byte;
    ownership: Byte;
    materialized: Byte;
    state: Byte;
    generation: UInt64;
  end;

  PTensorDescriptorC = ^TTensorDescriptorC;

  { ========================================================================
    Result Structure
    ======================================================================== }

  TTensorResultC = packed record
    success: Int32;
    error_code: UInt32;
    error_msg: array[0..255] of AnsiChar;
  end;

  { ========================================================================
    Operation Result
    ======================================================================== }

  TTensorOpResultC = packed record
    descriptor: TTensorDescriptorC;
    result: TTensorResultC;
  end;

  { ========================================================================
    Ownership State Enumeration
    ======================================================================== }

  TOwnershipState = (
    OWNERSHIP_OWNED = 0,
    OWNERSHIP_SHARED = 1,
    OWNERSHIP_EXCLUSIVE = 2
  );

  { ========================================================================
    Governance State Enumeration
    ======================================================================== }

  TGovernanceState = (
    GOVERNANCE_SHARED = 0,      { B=0: shared/non-canonical }
    GOVERNANCE_CANONICAL = 1    { B=1: canonical/contiguous }
  );

{ ============================================================================
  FFI Declarations — map to Rust tensor_ffi.rs
  ============================================================================ }

{ External C function declarations }

function tensor_allocate(
  rank: UInt32;
  shape: PUInt64;
  element_size: UInt64
): TTensorOpResultC; cdecl; external;

function tensor_deallocate(
  desc: PTensorDescriptorC
): TTensorResultC; cdecl; external;

function tensor_view(
  source: PTensorDescriptorC;
  start_indices: PUInt64;
  end_indices: PUInt64
): TTensorOpResultC; cdecl; external;

function tensor_materialize(
  source: PTensorDescriptorC
): TTensorOpResultC; cdecl; external;

function tensor_validate_generation(
  desc: PTensorDescriptorC
): TTensorResultC; cdecl; external;

function tensor_load(
  desc: PTensorDescriptorC;
  indices: PUInt64;
  out_value: PSingle
): TTensorResultC; cdecl; external;

function tensor_store(
  desc: PTensorDescriptorC;
  indices: PUInt64;
  value: Single
): TTensorResultC; cdecl; external;

function tensor_info(
  desc: PTensorDescriptorC;
  out_buffer: PByte;
  out_len: SizeInt
): TTensorResultC; cdecl; external;

{ ============================================================================
  Pascal Wrapper Functions (idiomatic error handling)
  ============================================================================ }

type
  EFTensorError = class(Exception);

{ Allocate tensor, raise exception on failure }
procedure FTensorAllocate(
  out Desc: TTensorDescriptorC;
  Rank: UInt32;
  const Shape: array of UInt64;
  ElementSize: UInt64 = 4
);

{ Deallocate tensor, raise exception on failure }
procedure FTensorDeallocate(const Desc: TTensorDescriptorC);

{ Create view, raise exception on failure }
procedure FTensorView(
  out ViewDesc: TTensorDescriptorC;
  const Source: TTensorDescriptorC;
  const StartIndices, EndIndices: array of UInt64
);

{ Materialize view, raise exception on failure }
procedure FTensorMaterialize(
  out MatDesc: TTensorDescriptorC;
  const Source: TTensorDescriptorC
);

{ Validate generation (stale descriptor check) }
function FTensorValidateGeneration(const Desc: TTensorDescriptorC): Boolean;

{ Load element value }
function FTensorLoad(
  const Desc: TTensorDescriptorC;
  const Indices: array of UInt64
): Single;

{ Store element value }
procedure FTensorStore(
  const Desc: TTensorDescriptorC;
  const Indices: array of UInt64;
  Value: Single
);

{ Get info string (for debugging) }
function FTensorInfo(const Desc: TTensorDescriptorC): String;

{ Helper: check result and raise on error }
procedure CheckFTensorResult(const Result: TTensorResultC; const Context: String);

implementation

procedure CheckFTensorResult(const Result: TTensorResultC; const Context: String);
var
  ErrorMsg: String;
begin
  if Result.success = 0 then
  begin
    SetLength(ErrorMsg, 256);
    ErrorMsg := PAnsiChar(@Result.error_msg[0]);
    raise EFTensorError.Create(
      Format('[%s] Error %d: %s', [Context, Result.error_code, ErrorMsg])
    );
  end;
end;

procedure FTensorAllocate(
  out Desc: TTensorDescriptorC;
  Rank: UInt32;
  const Shape: array of UInt64;
  ElementSize: UInt64 = 4
);
var
  OpResult: TTensorOpResultC;
  ShapeArray: array of UInt64;
  I: Integer;
begin
  if Rank = 0 then
    raise EFTensorError.Create('Rank must be > 0');

  if Rank <> High(Shape) + 1 then
    raise EFTensorError.Create('Shape array length mismatch');

  { Copy shape to heap }
  SetLength(ShapeArray, Rank);
  for I := Low(Shape) to High(Shape) do
    ShapeArray[I] := Shape[I];

  { Call FFI }
  OpResult := tensor_allocate(Rank, @ShapeArray[0], ElementSize);
  CheckFTensorResult(OpResult.result, 'tensor_allocate');

  Desc := OpResult.descriptor;
end;

procedure FTensorDeallocate(const Desc: TTensorDescriptorC);
var
  LocalDesc: TTensorDescriptorC;
  Result: TTensorResultC;
begin
  LocalDesc := Desc;
  Result := tensor_deallocate(@LocalDesc);
  CheckFTensorResult(Result, 'tensor_deallocate');
end;

procedure FTensorView(
  out ViewDesc: TTensorDescriptorC;
  const Source: TTensorDescriptorC;
  const StartIndices, EndIndices: array of UInt64
);
var
  OpResult: TTensorOpResultC;
  SrcCopy: TTensorDescriptorC;
  StartArray, EndArray: array of UInt64;
  I: Integer;
begin
  if High(StartIndices) <> High(EndIndices) then
    raise EFTensorError.Create('Index array length mismatch');

  if High(StartIndices) + 1 <> Source.rank then
    raise EFTensorError.Create('Index count mismatch with tensor rank');

  SetLength(StartArray, Source.rank);
  SetLength(EndArray, Source.rank);

  for I := Low(StartIndices) to High(StartIndices) do
  begin
    StartArray[I] := StartIndices[I];
    EndArray[I] := EndIndices[I];
  end;

  SrcCopy := Source;
  OpResult := tensor_view(@SrcCopy, @StartArray[0], @EndArray[0]);
  CheckFTensorResult(OpResult.result, 'tensor_view');

  ViewDesc := OpResult.descriptor;
end;

procedure FTensorMaterialize(
  out MatDesc: TTensorDescriptorC;
  const Source: TTensorDescriptorC
);
var
  OpResult: TTensorOpResultC;
  SrcCopy: TTensorDescriptorC;
begin
  SrcCopy := Source;
  OpResult := tensor_materialize(@SrcCopy);
  CheckFTensorResult(OpResult.result, 'tensor_materialize');

  MatDesc := OpResult.descriptor;
end;

function FTensorValidateGeneration(const Desc: TTensorDescriptorC): Boolean;
var
  LocalDesc: TTensorDescriptorC;
  Result: TTensorResultC;
begin
  LocalDesc := Desc;
  Result := tensor_validate_generation(@LocalDesc);
  Exit(Result.success <> 0);
end;

function FTensorLoad(
  const Desc: TTensorDescriptorC;
  const Indices: array of UInt64
): Single;
var
  LocalDesc: TTensorDescriptorC;
  IndicesArray: array of UInt64;
  Value: Single;
  Result: TTensorResultC;
  I: Integer;
begin
  if High(Indices) + 1 <> Desc.rank then
    raise EFTensorError.Create('Index count mismatch');

  SetLength(IndicesArray, Desc.rank);
  for I := Low(Indices) to High(Indices) do
    IndicesArray[I] := Indices[I];

  LocalDesc := Desc;
  Result := tensor_load(@LocalDesc, @IndicesArray[0], @Value);
  CheckFTensorResult(Result, 'tensor_load');

  Exit(Value);
end;

procedure FTensorStore(
  const Desc: TTensorDescriptorC;
  const Indices: array of UInt64;
  Value: Single
);
var
  LocalDesc: TTensorDescriptorC;
  IndicesArray: array of UInt64;
  Result: TTensorResultC;
  I: Integer;
begin
  if High(Indices) + 1 <> Desc.rank then
    raise EFTensorError.Create('Index count mismatch');

  SetLength(IndicesArray, Desc.rank);
  for I := Low(Indices) to High(Indices) do
    IndicesArray[I] := Indices[I];

  LocalDesc := Desc;
  Result := tensor_store(@LocalDesc, @IndicesArray[0], Value);
  CheckFTensorResult(Result, 'tensor_store');
end;

function FTensorInfo(const Desc: TTensorDescriptorC): String;
var
  LocalDesc: TTensorDescriptorC;
  Buffer: array[0..511] of AnsiChar;
  Result: TTensorResultC;
begin
  FillChar(Buffer, SizeOf(Buffer), 0);
  LocalDesc := Desc;
  Result := tensor_info(@LocalDesc, PByte(@Buffer), SizeOf(Buffer));
  CheckFTensorResult(Result, 'tensor_info');
  Exit(String(PAnsiChar(@Buffer)));
end;

end.
