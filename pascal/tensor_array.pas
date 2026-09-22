{ TensorArray: Array/buffer management with Arc-like reference counting.
  Mimics Rust Arc<Vec<T>> semantics using explicit reference counting.

  Semantic invariants:
    - Physical storage is strictly contiguous 1D array
    - Reference counting prevents use-after-free
    - Clone operation increments refcount without copying data
    - Drop operation decrements refcount, frees when reaching 0
}

unit tensor_array;

{$MODE FPC}

interface

uses
  buffer;

type
  { Arc-wrapped buffer }
  TTensorArray = record
    { Underlying buffer with refcount }
    buffer: PBuffer;

    { Current generation of this reference }
    generation: uint32;
  end;

  PTensorArray = ^TTensorArray;

{ Create new tensor array with given size }
function CreateTensorArray(size: uint32): TTensorArray;

{ Clone a tensor array (increment refcount, share buffer) }
function CloneTensorArray(const arr: TTensorArray): TTensorArray;

{ Drop a tensor array (decrement refcount) }
procedure DropTensorArray(var arr: TTensorArray);

{ Get element at index (with bounds checking) }
function GetElement(const arr: TTensorArray; index: uint32): single;

{ Set element at index (requires exclusive ownership) }
procedure SetElement(var arr: TTensorArray; index: uint32; value: single);

{ Get pointer to buffer data }
function GetDataPtr(const arr: TTensorArray): ^single;

{ Get size of tensor array }
function GetSize(const arr: TTensorArray): uint32;

{ Check if exclusively owned }
function IsExclusivelyOwned(const arr: TTensorArray): boolean;

{ Check if refcount > 1 (shared) }
function IsShared(const arr: TTensorArray): boolean;

{ Validate array consistency }
function IsValidArray(const arr: TTensorArray): boolean;

{ Clone on write: if refcount > 1, allocate new buffer }
procedure CopyOnWrite(var arr: TTensorArray);

{ Get reference count }
function GetRefCount(const arr: TTensorArray): uint32;

implementation

function CreateTensorArray(size: uint32): TTensorArray;
var
  buf: PBuffer;
begin
  buf := AllocateBuffer(size);
  if buf = nil then
  begin
    result.buffer := nil;
    result.generation := 0;
  end
  else
  begin
    result.buffer := buf;
    result.generation := buf^.generation;
  end;
end;

function CloneTensorArray(const arr: TTensorArray): TTensorArray;
var
  buf: PBuffer;
begin
  if arr.buffer = nil then
  begin
    result.buffer := nil;
    result.generation := 0;
  end
  else
  begin
    buf := arr.buffer;
    CloneBuffer(buf);
    result.buffer := buf;
    result.generation := buf^.generation;
  end;
end;

procedure DropTensorArray(var arr: TTensorArray);
begin
  ReleaseBuffer(arr.buffer);
  arr.generation := 0;
end;

function GetElement(const arr: TTensorArray; index: uint32): single;
begin
  if (arr.buffer = nil) or (index >= arr.buffer^.size) then
    result := 0.0
  else
    result := arr.buffer^.data^[index];
end;

procedure SetElement(var arr: TTensorArray; index: uint32; value: single);
begin
  if (arr.buffer <> nil) and (index < arr.buffer^.size) then
  begin
    arr.buffer^.data^[index] := value;
    arr.generation := arr.buffer^.generation;
  end;
end;

function GetDataPtr(const arr: TTensorArray): ^single;
begin
  if arr.buffer = nil then
    result := nil
  else
    result := arr.buffer^.data;
end;

function GetSize(const arr: TTensorArray): uint32;
begin
  if arr.buffer = nil then
    result := 0
  else
    result := arr.buffer^.size;
end;

function IsExclusivelyOwned(const arr: TTensorArray): boolean;
begin
  if arr.buffer = nil then
    result := false
  else
    result := IsExclusivelyOwned(arr.buffer);
end;

function IsShared(const arr: TTensorArray): boolean;
begin
  if arr.buffer = nil then
    result := false
  else
    result := (arr.buffer^.refcount > 1);
end;

function IsValidArray(const arr: TTensorArray): boolean;
begin
  if arr.buffer = nil then
    result := false
  else
    result := (arr.buffer^.size > 0) and
              (arr.buffer^.data <> nil) and
              (arr.generation = arr.buffer^.generation);
end;

procedure CopyOnWrite(var arr: TTensorArray);
var
  new_buf: PBuffer;
begin
  if arr.buffer = nil then
    exit;

  if arr.buffer^.refcount <= 1 then
    exit; { Already exclusive }

  { Allocate new buffer and copy data }
  new_buf := AllocateBuffer(arr.buffer^.size);
  if new_buf = nil then
    exit;

  CopyBufferData(arr.buffer, 0, new_buf, 0, arr.buffer^.size);

  { Release old buffer }
  ReleaseBuffer(arr.buffer);

  { Adopt new buffer }
  arr.buffer := new_buf;
  arr.generation := new_buf^.generation;
end;

function GetRefCount(const arr: TTensorArray): uint32;
begin
  if arr.buffer = nil then
    result := 0
  else
    result := arr.buffer^.refcount;
end;

end.
