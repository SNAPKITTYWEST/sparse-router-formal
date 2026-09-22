{ TensorDescriptor: Packed record for C-compatible tensor metadata
  Represents the semantic view layer: Shape, Strides, Offset.
  Implements B_layout and B_own governance flags for binary semantics.

  Mathematical Definition:
    - Shape S = (s_0, ..., s_{d-1}): Logical dimensions
    - Strides Σ = (σ_0, ..., σ_{d-1}): Step sizes along each axis
    - Offset O: Starting position in physical buffer
    - B_layout: 1 iff strides are canonical (σ_i = ∏_{j>i} s_j)
    - B_own: 1 iff this view exclusively owns buffer (refcount=1)

  Index mapping: f(x_0, ..., x_{d-1}) = O + Σ(x_i * σ_i)
}

unit descriptor;

{$MODE FPC}
{$PACKRECORDS C}
{$PACKENUM ON}

interface

type
  { Packed C-compatible tensor descriptor }
  TTensorDescriptor = packed record
    { Rank (dimensionality) }
    rank: uint32;

    { Shape array: s_0, s_1, ..., s_{rank-1} }
    shape_ptr: ^uint32;

    { Strides: σ_0, σ_1, ..., σ_{rank-1} }
    strides_ptr: ^uint32;

    { Memory offset in physical buffer }
    offset: uint32;

    { Total number of elements: ∏ shape_i }
    len: uint32;

    { B_layout: canonical strides (1 if contiguous) }
    is_contiguous: uint8;

    { B_own: exclusive ownership (1 if owns_storage) }
    owns_storage: uint8;

    { Generation counter for temporal safety }
    generation: uint32;
  end;

{ Create a descriptor from rank and preallocated arrays }
function CreateDescriptor(
  rank: uint32;
  shape_ptr: ^uint32;
  strides_ptr: ^uint32
): TTensorDescriptor;

{ Check if descriptor is valid (non-zero rank, non-nil pointers) }
function IsValidDescriptor(const desc: TTensorDescriptor): boolean;

{ Compute total length from shape }
function ComputeLength(const shape: array of uint32): uint32;

{ Compute canonical strides for given shape }
procedure ComputeCanonicalStrides(
  const shape: array of uint32;
  var strides: array of uint32
);

{ Check if strides are canonical for given shape }
function IsCanonicalStrides(
  const shape: array of uint32;
  const strides: array of uint32
): boolean;

{ Check descriptor equivalence (same shape, strides, offset) }
function DescriptorsEqual(const a, b: TTensorDescriptor): boolean;

implementation

function CreateDescriptor(
  rank: uint32;
  shape_ptr: ^uint32;
  strides_ptr: ^uint32
): TTensorDescriptor;
var
  len, i, prod: uint32;
begin
  len := 1;
  for i := 0 to rank - 1 do
    len := len * shape_ptr^[i];

  with result do
  begin
    descriptor.rank := rank;
    descriptor.shape_ptr := shape_ptr;
    descriptor.strides_ptr := strides_ptr;
    descriptor.offset := 0;
    descriptor.len := len;
    descriptor.is_contiguous := 0;
    descriptor.owns_storage := 0;
    descriptor.generation := 0;
  end;
end;

function IsValidDescriptor(const desc: TTensorDescriptor): boolean;
begin
  result := (desc.rank > 0) and
            (desc.shape_ptr <> nil) and
            (desc.strides_ptr <> nil) and
            (desc.len > 0);
end;

function ComputeLength(const shape: array of uint32): uint32;
var
  i: uint32;
begin
  result := 1;
  for i := 0 to high(shape) do
    result := result * shape[i];
end;

procedure ComputeCanonicalStrides(
  const shape: array of uint32;
  var strides: array of uint32
);
var
  i: integer;
  stride: uint32;
begin
  stride := 1;
  for i := high(shape) downto 0 do
  begin
    strides[i] := stride;
    stride := stride * shape[i];
  end;
end;

function IsCanonicalStrides(
  const shape: array of uint32;
  const strides: array of uint32
): boolean;
var
  i, stride: uint32;
begin
  stride := 1;
  for i := high(shape) downto 0 do
  begin
    if strides[i] <> stride then
      exit(false);
    stride := stride * shape[i];
  end;
  result := true;
end;

function DescriptorsEqual(const a, b: TTensorDescriptor): boolean;
var
  i: uint32;
begin
  if (a.rank <> b.rank) or
     (a.offset <> b.offset) or
     (a.len <> b.len) then
    exit(false);

  for i := 0 to a.rank - 1 do
  begin
    if a.shape_ptr^[i] <> b.shape_ptr^[i] then
      exit(false);
    if a.strides_ptr^[i] <> b.strides_ptr^[i] then
      exit(false);
  end;

  result := true;
end;

end.
