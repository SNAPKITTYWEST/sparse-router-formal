{ NraayTensor Reference Implementation: Main tensor operations.
  Implements core contract operations: ALLOC, FREE, VIEW, SLICE, TRANSPOSE, RESHAPE, MATERIALIZE

  Semantic layer:
    - Physical Storage: Contiguous buffer managed by tensor_array (Arc semantics)
    - Semantic View: Descriptor (shape, strides, offset, governance flags)
    - Ownership: Token tracking OWNED/SHARED/EXCLUSIVE state
}

unit tensor;

{$MODE FPC}

interface

uses
  descriptor, buffer, tensor_array, ownership, materialize;

type
  { NraayTensor: Semantic view + physical storage }
  TNraayTensor = record
    { Physical buffer (Arc-wrapped) }
    data: TTensorArray;

    { Semantic descriptor }
    descriptor: TTensorDescriptor;

    { Ownership token }
    ownership: TOwnershipToken;

    { Shape array (must persist for descriptor) }
    shape: array of uint32;

    { Strides array (must persist for descriptor) }
    strides: array of uint32;
  end;

  { Operation result }
  TTensorResult = (
    TR_SUCCESS,
    TR_INVALID_SHAPE,
    TR_DIMENSION_MISMATCH,
    TR_OUT_OF_BOUNDS,
    TR_INVALID_PERMUTATION,
    TR_ALLOCATION_FAILED,
    TR_NULL_POINTER
  );

{ Create tensor with given shape (ALLOC operation) }
function CreateTensor(const shape: array of uint32): TNraayTensor;

{ Clone tensor (increment refcount, shared ownership) }
function CloneTensor(const t: TNraayTensor): TNraayTensor;

{ Destroy tensor (decrement refcount) }
procedure DestroyTensor(var t: TNraayTensor);

{ Get element at coordinate }
function GetElement(const t: TNraayTensor; const coord: array of uint32): single;

{ Set element at coordinate }
procedure SetElement(var t: TNraayTensor; const coord: array of uint32; value: single);

{ Virtual slicing: zero-copy view with ranges (start, end, step) }
function Slice(
  const t: TNraayTensor;
  const ranges: array of array of uint32
): TNraayTensor;

{ Transpose via permutation (axes rearrangement) }
function Transpose(
  const t: TNraayTensor;
  const perm: array of uint32
): TNraayTensor;

{ Materialize tensor (restore contiguity and exclusive ownership) }
procedure Materialize(var t: TNraayTensor);

{ Check if tensor is valid }
function IsValid(const t: TNraayTensor): boolean;

{ Get total element count }
function Length(const t: TNraayTensor): uint32;

{ Get rank }
function Rank(const t: TNraayTensor): uint32;

{ Compute canonical strides for shape }
procedure ComputeStrides(const shape: array of uint32; var strides: array of uint32);

{ Result code to string }
function ResultToString(result: TTensorResult): string;

implementation

procedure ComputeStrides(const shape: array of uint32; var strides: array of uint32);
var
  i, stride: uint32;
begin
  stride := 1;
  for i := high(shape) downto 0 do
  begin
    strides[i] := stride;
    stride := stride * shape[i];
  end;
end;

function CreateTensor(const shape: array of uint32): TNraayTensor;
var
  size: uint32;
  i: uint32;
begin
  setlength(result.shape, length(shape));
  setlength(result.strides, length(shape));

  { Copy shape }
  size := 1;
  for i := 0 to high(shape) do
  begin
    result.shape[i] := shape[i];
    size := size * shape[i];
  end;

  { Compute strides }
  ComputeStrides(result.shape, result.strides);

  { Allocate buffer }
  result.data := CreateTensorArray(size);

  { Create descriptor }
  with result.descriptor do
  begin
    rank := length(shape);
    shape_ptr := @result.shape[0];
    strides_ptr := @result.strides[0];
    offset := 0;
    len := size;
    is_contiguous := 1;
    owns_storage := 1;
    generation := 0;
  end;

  { Initialize ownership }
  result.ownership := CreateOwnershipToken(result.descriptor.generation);
end;

function CloneTensor(const t: TNraayTensor): TNraayTensor;
begin
  result := t;
  result.data := CloneTensorArray(t.data);
  result.ownership := TransitionToShared(t.ownership);
end;

procedure DestroyTensor(var t: TNraayTensor);
begin
  DropTensorArray(t.data);
  setlength(t.shape, 0);
  setlength(t.strides, 0);
end;

function GetElement(const t: TNraayTensor; const coord: array of uint32): single;
var
  i, offset: uint32;
begin
  if length(coord) <> t.descriptor.rank then
    exit(0.0);

  offset := t.descriptor.offset;
  for i := 0 to t.descriptor.rank - 1 do
  begin
    if coord[i] >= t.descriptor.shape_ptr^[i] then
      exit(0.0);
    offset := offset + coord[i] * t.descriptor.strides_ptr^[i];
  end;

  result := GetElement(t.data, offset);
end;

procedure SetElement(var t: TNraayTensor; const coord: array of uint32; value: single);
var
  i, offset: uint32;
begin
  if length(coord) <> t.descriptor.rank then
    exit;

  offset := t.descriptor.offset;
  for i := 0 to t.descriptor.rank - 1 do
  begin
    if coord[i] >= t.descriptor.shape_ptr^[i] then
      exit;
    offset := offset + coord[i] * t.descriptor.strides_ptr^[i];
  end;

  SetElement(t.data, offset, value);
end;

function Slice(
  const t: TNraayTensor;
  const ranges: array of array of uint32
): TNraayTensor;
var
  i, j: uint32;
  new_shape: array of uint32;
  new_strides: array of uint32;
  new_offset: uint32;
begin
  if length(ranges) <> t.descriptor.rank then
    exit(t); { Invalid dimensions }

  setlength(new_shape, t.descriptor.rank);
  setlength(new_strides, t.descriptor.rank);
  new_offset := t.descriptor.offset;

  for i := 0 to t.descriptor.rank - 1 do
  begin
    if length(ranges[i]) < 3 then
      exit(t); { Invalid range format }

    with ranges[i] do
    begin
      if (ranges[i][0] >= t.descriptor.shape_ptr^[i]) or
         (ranges[i][1] > t.descriptor.shape_ptr^[i]) or
         (ranges[i][0] >= ranges[i][1]) or
         (ranges[i][2] = 0) then
        exit(t); { Invalid slice }
    end;

    new_shape[i] := (ranges[i][1] - ranges[i][0] + ranges[i][2] - 1) div ranges[i][2];
    new_offset := new_offset + ranges[i][0] * t.descriptor.strides_ptr^[i];
    new_strides[i] := t.descriptor.strides_ptr^[i] * ranges[i][2];
  end;

  result := t;
  result.shape := new_shape;
  result.strides := new_strides;
  result.descriptor.shape_ptr := @result.shape[0];
  result.descriptor.strides_ptr := @result.strides[0];
  result.descriptor.offset := new_offset;
  result.descriptor.is_contiguous := 0;
  result.descriptor.owns_storage := 0;
  result.ownership := TransitionToShared(t.ownership);
end;

function Transpose(
  const t: TNraayTensor;
  const perm: array of uint32
): TNraayTensor;
var
  i: uint32;
  new_shape: array of uint32;
  new_strides: array of uint32;
begin
  if length(perm) <> t.descriptor.rank then
    exit(t);

  setlength(new_shape, t.descriptor.rank);
  setlength(new_strides, t.descriptor.rank);

  for i := 0 to t.descriptor.rank - 1 do
  begin
    if perm[i] >= t.descriptor.rank then
      exit(t); { Invalid permutation }
    new_shape[i] := t.descriptor.shape_ptr^[perm[i]];
    new_strides[i] := t.descriptor.strides_ptr^[perm[i]];
  end;

  result := t;
  result.shape := new_shape;
  result.strides := new_strides;
  result.descriptor.shape_ptr := @result.shape[0];
  result.descriptor.strides_ptr := @result.strides[0];
  result.descriptor.is_contiguous := 0;
end;

procedure Materialize(var t: TNraayTensor);
var
  mat_result: TMaterializationResult;
begin
  mat_result := MaterializeTensorArray(t.data, t.descriptor);
  if mat_result.success then
  begin
    { Reset strides to canonical }
    ComputeStrides(t.shape, t.strides);
    t.descriptor.offset := 0;
    t.descriptor.is_contiguous := 1;
    t.descriptor.owns_storage := 1;
    t.descriptor.generation := mat_result.new_generation;
    t.ownership := TransitionToExclusive(t.ownership, mat_result.new_generation);
  end;
end;

function IsValid(const t: TNraayTensor): boolean;
begin
  result := (GetSize(t.data) > 0) and
            (t.descriptor.rank > 0) and
            (t.descriptor.len > 0);
end;

function Length(const t: TNraayTensor): uint32;
begin
  result := t.descriptor.len;
end;

function Rank(const t: TNraayTensor): uint32;
begin
  result := t.descriptor.rank;
end;

function ResultToString(result: TTensorResult): string;
begin
  case result of
    TR_SUCCESS:
      ResultToString := 'SUCCESS';
    TR_INVALID_SHAPE:
      ResultToString := 'INVALID_SHAPE';
    TR_DIMENSION_MISMATCH:
      ResultToString := 'DIMENSION_MISMATCH';
    TR_OUT_OF_BOUNDS:
      ResultToString := 'OUT_OF_BOUNDS';
    TR_INVALID_PERMUTATION:
      ResultToString := 'INVALID_PERMUTATION';
    TR_ALLOCATION_FAILED:
      ResultToString := 'ALLOCATION_FAILED';
    TR_NULL_POINTER:
      ResultToString := 'NULL_POINTER';
  else
    ResultToString := 'UNKNOWN';
  end;
end;

end.
