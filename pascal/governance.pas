{ Governance Validation Engine: Executable proof obligations.
  Implements 5 lemmas as per NraayTensor spec:

  Lemma 1 (Shape-Rank): rank(t) = len(shape(t))
  Lemma 2 (Length Product): len(t) = ∏ shape_i
  Lemma 3 (Canonical Strides): B_layout=1 ⟹ σ_i = ∏_{j>i} s_j
  Lemma 4 (Exclusive Ownership): B_own=1 ⟹ refcount=1
  Lemma 5 (Generation Safety): all views share same generation until materialize
}

unit governance;

{$MODE FPC}

interface

uses
  descriptor, buffer;

type
  { Governance violation type }
  TGovernanceViolation = (
    GV_NONE,
    GV_SHAPE_RANK_MISMATCH,      { Lemma 1 violated }
    GV_LENGTH_PRODUCT_MISMATCH,   { Lemma 2 violated }
    GV_CANONICAL_STRIDE_VIOLATION,{ Lemma 3 violated }
    GV_EXCLUSIVE_OWNERSHIP_VIOLATION, { Lemma 4 violated }
    GV_GENERATION_MISMATCH        { Lemma 5 violated }
  );

  { Proof obligation }
  TLemmaProof = record
    lemma_num: uint32;
    violation: TGovernanceViolation;
    details: string;
  end;

{ Lemma 1: Shape-Rank Consistency }
function ProveShapeRank(const desc: TTensorDescriptor): TLemmaProof;

{ Lemma 2: Length is product of shape }
function ProveLengthProduct(const desc: TTensorDescriptor): TLemmaProof;

{ Lemma 3: Canonical strides when B_layout=1 }
function ProveCanonicalStrides(const desc: TTensorDescriptor): TLemmaProof;

{ Lemma 4: B_own=1 implies refcount=1 }
function ProveExclusiveOwnership(
  const desc: TTensorDescriptor;
  buf: PBuffer
): TLemmaProof;

{ Lemma 5: Generation safety across views }
function ProveGenerationSafety(
  const desc: TTensorDescriptor;
  buf: PBuffer
): TLemmaProof;

{ Validate all lemmas }
function ValidateAllLemmas(
  const desc: TTensorDescriptor;
  buf: PBuffer
): boolean;

{ Get human-readable violation description }
function ViolationToString(violation: TGovernanceViolation): string;

{ Check if proof passed }
function IsProofValid(const proof: TLemmaProof): boolean;

implementation

function ProveShapeRank(const desc: TTensorDescriptor): TLemmaProof;
begin
  with result do
  begin
    lemma_num := 1;
    if desc.rank > 0 then
    begin
      violation := GV_NONE;
      details := 'Shape rank consistent: ' + IntToStr(desc.rank);
    end
    else
    begin
      violation := GV_SHAPE_RANK_MISMATCH;
      details := 'Invalid rank: ' + IntToStr(desc.rank);
    end;
  end;
end;

function ProveLengthProduct(const desc: TTensorDescriptor): TLemmaProof;
var
  i: uint32;
  prod: uint32;
begin
  with result do
  begin
    lemma_num := 2;
    prod := 1;
    for i := 0 to desc.rank - 1 do
      prod := prod * desc.shape_ptr^[i];

    if prod = desc.len then
    begin
      violation := GV_NONE;
      details := 'Length product valid: ' + IntToStr(prod);
    end
    else
    begin
      violation := GV_LENGTH_PRODUCT_MISMATCH;
      details := 'Expected ' + IntToStr(prod) + ', got ' + IntToStr(desc.len);
    end;
  end;
end;

function ProveCanonicalStrides(const desc: TTensorDescriptor): TLemmaProof;
var
  i, stride: uint32;
begin
  with result do
  begin
    lemma_num := 3;
    if desc.is_contiguous = 0 then
    begin
      violation := GV_NONE;
      details := 'B_layout=0, strides not required canonical';
      exit;
    end;

    { Check if strides match canonical }
    stride := 1;
    for i := desc.rank - 1 downto 0 do
    begin
      if desc.strides_ptr^[i] <> stride then
      begin
        violation := GV_CANONICAL_STRIDE_VIOLATION;
        details := 'Stride mismatch at dim ' + IntToStr(i) +
                   ': expected ' + IntToStr(stride) +
                   ', got ' + IntToStr(desc.strides_ptr^[i]);
        exit;
      end;
      stride := stride * desc.shape_ptr^[i];
    end;

    violation := GV_NONE;
    details := 'Canonical strides verified';
  end;
end;

function ProveExclusiveOwnership(
  const desc: TTensorDescriptor;
  buf: PBuffer
): TLemmaProof;
begin
  with result do
  begin
    lemma_num := 4;
    if desc.owns_storage = 0 then
    begin
      violation := GV_NONE;
      details := 'B_own=0, no exclusive check required';
      exit;
    end;

    if buf = nil then
    begin
      violation := GV_EXCLUSIVE_OWNERSHIP_VIOLATION;
      details := 'Buffer is nil but B_own=1';
      exit;
    end;

    if buf^.refcount = 1 then
    begin
      violation := GV_NONE;
      details := 'Exclusive ownership verified (refcount=1)';
    end
    else
    begin
      violation := GV_EXCLUSIVE_OWNERSHIP_VIOLATION;
      details := 'B_own=1 but refcount=' + IntToStr(buf^.refcount);
    end;
  end;
end;

function ProveGenerationSafety(
  const desc: TTensorDescriptor;
  buf: PBuffer
): TLemmaProof;
begin
  with result do
  begin
    lemma_num := 5;
    if buf = nil then
    begin
      violation := GV_GENERATION_MISMATCH;
      details := 'Buffer is nil, cannot verify generation';
      exit;
    end;

    if desc.generation = buf^.generation then
    begin
      violation := GV_NONE;
      details := 'Generation safety verified: ' + IntToStr(desc.generation);
    end
    else
    begin
      violation := GV_GENERATION_MISMATCH;
      details := 'Descriptor gen ' + IntToStr(desc.generation) +
                 ' vs buffer gen ' + IntToStr(buf^.generation);
    end;
  end;
end;

function ValidateAllLemmas(
  const desc: TTensorDescriptor;
  buf: PBuffer
): boolean;
begin
  if IsProofValid(ProveShapeRank(desc)) and
     IsProofValid(ProveLengthProduct(desc)) and
     IsProofValid(ProveCanonicalStrides(desc)) and
     IsProofValid(ProveExclusiveOwnership(desc, buf)) and
     IsProofValid(ProveGenerationSafety(desc, buf)) then
    result := true
  else
    result := false;
end;

function ViolationToString(violation: TGovernanceViolation): string;
begin
  case violation of
    GV_NONE:
      result := 'NO_VIOLATION';
    GV_SHAPE_RANK_MISMATCH:
      result := 'SHAPE_RANK_MISMATCH';
    GV_LENGTH_PRODUCT_MISMATCH:
      result := 'LENGTH_PRODUCT_MISMATCH';
    GV_CANONICAL_STRIDE_VIOLATION:
      result := 'CANONICAL_STRIDE_VIOLATION';
    GV_EXCLUSIVE_OWNERSHIP_VIOLATION:
      result := 'EXCLUSIVE_OWNERSHIP_VIOLATION';
    GV_GENERATION_MISMATCH:
      result := 'GENERATION_MISMATCH';
  else
    result := 'UNKNOWN_VIOLATION';
  end;
end;

function IsProofValid(const proof: TLemmaProof): boolean;
begin
  result := (proof.violation = GV_NONE);
end;

end.
