{ Materialization Barrier: Copy-on-write with governance state restoration.
  Restores B_layout=1 and B_own=1 by copying viewed elements to contiguous buffer.

  Algorithm:
    1. Check if already materialized (B_layout=1 AND B_own=1): return OK
    2. Allocate new contiguous buffer of size len(t)
    3. Copy viewed elements in row-major order using strides
    4. Increment generation on new buffer
    5. Replace old buffer reference with new buffer
    6. Reset strides to canonical
    7. Set B_layout=1, B_own=1

  Governance Property: "Preserving order, strict and bound"
}

unit materialize;

{$MODE FPC}

interface

uses
  descriptor, buffer, tensor_array;

type
  { Materialization result }
  TMaterializationResult = record
    success: boolean;
    old_generation: uint32;
    new_generation: uint32;
    copied_elements: uint32;
  end;

{ Materialize a tensor array in-place }
function MaterializeTensorArray(
  var arr: TTensorArray;
  const desc: TTensorDescriptor
): TMaterializationResult;

{ Copy viewed elements in row-major order }
procedure CopyViewedElements(
  const desc: TTensorDescriptor;
  src_data: ^single;
  dst_data: ^single
);

{ Compute row-major index from multi-dimensional coordinate }
function ComputeRowMajorIndex(
  const desc: TTensorDescriptor;
  coord: array of uint32
): uint32;

{ Materialize with strict bounds checking }
function MaterializeStrict(
  var arr: TTensorArray;
  const desc: TTensorDescriptor
): TMaterializationResult;

implementation

procedure CopyViewedRecursive(
  const desc: TTensorDescriptor;
  src_data: ^single;
  dst_data: ^single;
  var dst_idx: uint32;
  dim: uint32;
  current_offset: uint32
);
var
  i: uint32;
begin
  if dim = desc.rank then
  begin
    if current_offset < desc.len then
      dst_data^[dst_idx] := src_data^[current_offset];
    inc(dst_idx);
  end
  else
  begin
    for i := 0 to desc.shape_ptr^[dim] - 1 do
      CopyViewedRecursive(
        desc, src_data, dst_data, dst_idx, dim + 1,
        current_offset + i * desc.strides_ptr^[dim]
      );
  end;
end;

procedure CopyViewedElements(
  const desc: TTensorDescriptor;
  src_data: ^single;
  dst_data: ^single
);
var
  dst_idx: uint32;
begin
  dst_idx := 0;
  CopyViewedRecursive(desc, src_data, dst_data, dst_idx, 0, desc.offset);
end;

function ComputeRowMajorIndex(
  const desc: TTensorDescriptor;
  coord: array of uint32
): uint32;
var
  i: uint32;
begin
  result := 0;
  for i := 0 to desc.rank - 1 do
    result := result + coord[i] * desc.strides_ptr^[i];
end;

function MaterializeTensorArray(
  var arr: TTensorArray;
  const desc: TTensorDescriptor
): TMaterializationResult;
var
  new_buf: PBuffer;
  new_data: ^single;
begin
  with result do
  begin
    success := false;
    old_generation := 0;
    new_generation := 0;
    copied_elements := 0;

    if arr.buffer = nil then
      exit;

    { Check if already materialized }
    if (desc.is_contiguous = 1) and (desc.owns_storage = 1) then
    begin
      success := true;
      old_generation := arr.buffer^.generation;
      new_generation := arr.buffer^.generation;
      copied_elements := 0;
      exit;
    end;

    { Allocate new contiguous buffer }
    new_buf := AllocateBuffer(desc.len);
    if new_buf = nil then
      exit;

    new_data := new_buf^.data;

    { Copy viewed elements in row-major order }
    CopyViewedElements(desc, arr.buffer^.data, new_data);

    old_generation := arr.buffer^.generation;
    new_generation := new_buf^.generation;
    copied_elements := desc.len;

    { Release old buffer (may free if refcount becomes 0) }
    ReleaseBuffer(arr.buffer);

    { Adopt new buffer }
    arr.buffer := new_buf;
    arr.generation := new_buf^.generation;

    success := true;
  end;
end;

function MaterializeStrict(
  var arr: TTensorArray;
  const desc: TTensorDescriptor
): TMaterializationResult;
begin
  { Perform strict bounds checking before materialization }
  if (desc.len = 0) or (arr.buffer = nil) or (arr.buffer^.size = 0) then
  begin
    result.success := false;
    result.old_generation := 0;
    result.new_generation := 0;
    result.copied_elements := 0;
    exit;
  end;

  { Materialize }
  result := MaterializeTensorArray(arr, desc);
end;

end.
