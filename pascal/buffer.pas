{ Buffer: Physical storage management with generation tracking.
  Implements allocation, reference counting, and generation counters
  for temporal safety (use-after-free detection).

  Key properties:
    - Generational tracking: each buffer has monotonically increasing generation
    - Arc semantics: refcount > 1 indicates shared ownership
    - Memory safety: generation validation prevents stale references
}

unit buffer;

{$MODE FPC}

interface

type
  { Buffer metadata }
  TBuffer = record
    { Allocated data }
    data: ^single;

    { Physical size in elements }
    size: uint32;

    { Reference count (Arc semantics) }
    refcount: uint32;

    { Generation counter for temporal safety }
    generation: uint32;
  end;

  PBuffer = ^TBuffer;

{ Allocate a new buffer with given size }
function AllocateBuffer(size: uint32): PBuffer;

{ Free a buffer (when refcount reaches 0) }
procedure FreeBuffer(var buf: PBuffer);

{ Increment reference count (Arc::clone equivalent) }
procedure CloneBuffer(buf: PBuffer);

{ Decrement reference count }
procedure ReleaseBuffer(var buf: PBuffer);

{ Check if buffer is exclusively owned (refcount = 1) }
function IsExclusivelyOwned(buf: PBuffer): boolean;

{ Get current generation of buffer }
function GetGeneration(buf: PBuffer): uint32;

{ Increment generation on mutation }
procedure IncrementGeneration(buf: PBuffer);

{ Validate that generation matches expected }
function ValidateGeneration(buf: PBuffer; expected: uint32): boolean;

{ Zero-initialize a buffer region }
procedure ZeroBuffer(buf: PBuffer; size: uint32);

{ Copy data from source to destination buffer }
procedure CopyBufferData(
  src: PBuffer; src_offset: uint32;
  dst: PBuffer; dst_offset: uint32;
  count: uint32
);

implementation

var
  { Global generation counter for allocation tracking }
  global_generation: uint32 = 0;

function AllocateBuffer(size: uint32): PBuffer;
var
  buf: PBuffer;
begin
  new(buf);
  if buf = nil then
    exit(nil);

  getmem(buf^.data, size * sizeof(single));
  if buf^.data = nil then
  begin
    dispose(buf);
    exit(nil);
  end;

  buf^.size := size;
  buf^.refcount := 1;
  buf^.generation := global_generation;
  inc(global_generation);

  { Zero-initialize }
  ZeroBuffer(buf, size);
  result := buf;
end;

procedure FreeBuffer(var buf: PBuffer);
begin
  if buf = nil then
    exit;

  if buf^.data <> nil then
    freemem(buf^.data);

  dispose(buf);
  buf := nil;
end;

procedure CloneBuffer(buf: PBuffer);
begin
  if buf <> nil then
    inc(buf^.refcount);
end;

procedure ReleaseBuffer(var buf: PBuffer);
begin
  if buf = nil then
    exit;

  dec(buf^.refcount);
  if buf^.refcount = 0 then
    FreeBuffer(buf)
  else
    buf := nil;
end;

function IsExclusivelyOwned(buf: PBuffer): boolean;
begin
  if buf = nil then
    result := false
  else
    result := (buf^.refcount = 1);
end;

function GetGeneration(buf: PBuffer): uint32;
begin
  if buf = nil then
    result := 0
  else
    result := buf^.generation;
end;

procedure IncrementGeneration(buf: PBuffer);
begin
  if buf <> nil then
  begin
    inc(buf^.generation);
    inc(global_generation);
  end;
end;

function ValidateGeneration(buf: PBuffer; expected: uint32): boolean;
begin
  if buf = nil then
    result := false
  else
    result := (buf^.generation = expected);
end;

procedure ZeroBuffer(buf: PBuffer; size: uint32);
var
  i: uint32;
begin
  if buf = nil then
    exit;

  for i := 0 to size - 1 do
    buf^.data^[i] := 0.0;
end;

procedure CopyBufferData(
  src: PBuffer; src_offset: uint32;
  dst: PBuffer; dst_offset: uint32;
  count: uint32
);
var
  i: uint32;
begin
  if (src = nil) or (dst = nil) then
    exit;

  for i := 0 to count - 1 do
    dst^.data^[dst_offset + i] := src^.data^[src_offset + i];
end;

end.
