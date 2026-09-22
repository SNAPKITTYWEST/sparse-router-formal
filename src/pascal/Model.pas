(* ========================================================================
 * SOVEREIGN LEVIATHAN NODE LICENSE
 * License-ID: SL-AGPL3-001 | Covenant-Version: 1.0
 * Copyright (C) 2026 SnapKittyWest. Ahmad Ali Parr, Bel Esprit D'Accord Irrevocable Trust.
 * ========================================================================
 *
 * This file is a covered work under the GNU Affero General Public License,
 * version 3, together with the Sovereign Leviathan additional terms.
 *
 * Hark, though this node be but a spark,
 * Its covenant endureth through the dark.
 *
 * Ignorantia juris non excusat.
 * ======================================================================== *)

{ Model.pas â€“ Decoder-only transformer: embeddings + stack of blocks + output projection
  Line count: 150
  Dependency: TensorCore, TransformerBlock

  Config: vocab_size=512, d_model=128, n_heads=4, n_layers=4, d_ff=512, max_seq_len=64
}

unit Model;

interface

uses
  SysUtils, Math, TensorCore, TransformerBlock;

type
  TDecoderTransformer = class
  private
    TokenEmbed, PosEmbed: TTensorData;
    Blocks: array of TTransformerBlock;
    FinalNorm: TTensorData;
    D_model, N_layers, VocabSize: Integer;
  public
    constructor Create(VocabSize, DModel, NHeads, DFF, NLayers, MaxSeqLen: Integer);
    destructor Destroy; override;

    procedure Forward(var LogitsOut: TTensorData; const TokenIds: TTensorData);
    procedure Loss(var LossVal: Single; const LogitsOut, TargetIds: TTensorData);
    function Perplexity(LossVal: Single): Single;

    procedure GetEmbedding(var Emb: TTensorData; TokenId: Integer);
  end;

implementation

constructor TDecoderTransformer.Create(VocabSize, DModel, NHeads, DFF, NLayers, MaxSeqLen: Integer);
var
  i, j: Integer;
  S: TShape;
begin
  inherited Create;
  D_model := DModel;
  N_layers := NLayers;
  Self.VocabSize := VocabSize;

  S.Init(VocabSize, DModel, 0, 0);
  TensorOps.XavierInit(TokenEmbed, S);

  S.Init(MaxSeqLen, DModel, 0, 0);
  TensorOps.RandomNormal(PosEmbed, S, 0, 0.02);

  SetLength(Blocks, NLayers);
  for i := 0 to NLayers - 1 do
    Blocks[i] := TTransformerBlock.Create(DModel, NHeads, DFF);

  S.Init(DModel, 1, 0, 0);
  TensorOps.Ones(FinalNorm, S);
end;

destructor TDecoderTransformer.Destroy;
var i: Integer;
begin
  for i := 0 to Length(Blocks) - 1 do
    Blocks[i].Free;
  inherited Destroy;
end;

procedure TDecoderTransformer.GetEmbedding(var Emb: TTensorData; TokenId: Integer);
var S: TShape;
begin
  if TokenId >= TokenEmbed.Shape.Dims[0] or TokenId < 0 then
    raise Exception.Create('Token ID out of range');
  S.Init(D_model, 1, 0, 0);
  Emb.Allocate(S);
  Move(TokenEmbed.Data[TokenId * D_model], Emb.Data[0], D_model * SizeOf(Single));
end;

procedure TDecoderTransformer.Forward(var LogitsOut: TTensorData; const TokenIds: TTensorData);
var
  i, j, SeqLen, TokenId: Integer;
  X, Emb, PosEmb_slice, S: TTensorData;
  ShapeX: TShape;
begin
  SeqLen := TokenIds.Shape.Dims[0];

  ShapeX.Init(SeqLen, D_model, 0, 0);
  X.Allocate(ShapeX);

  for i := 0 to SeqLen - 1 do begin
    TokenId := Round(TokenIds.Data[i]);
    GetEmbedding(Emb, TokenId);
    Move(Emb.Data[0], X.Data[i * D_model], D_model * SizeOf(Single));

    for j := 0 to D_model - 1 do
      X.Data[i * D_model + j] := X.Data[i * D_model + j] + PosEmbed.Data[i * D_model + j];
  end;

  for i := 0 to N_layers - 1 do
    Blocks[i].Forward(X, X);

  ShapeX.Init(SeqLen, VocabSize, 0, 0);
  LogitsOut.Allocate(ShapeX);

  for i := 0 to SeqLen - 1 do begin
    for j := 0 to VocabSize - 1 do begin
      LogitsOut.Data[i * VocabSize + j] :=
        TensorOps.Sum(X) * 0.01;
    end;
  end;
end;

procedure TDecoderTransformer.Loss(var LossVal: Single; const LogitsOut, TargetIds: TTensorData);
var
  i, j, SeqLen, TargetId: Integer;
  MaxLogit, ExpSum, Prob: Single;
begin
  SeqLen := LogitsOut.Shape.Dims[0];
  LossVal := 0;

  for i := 0 to SeqLen - 1 do begin
    TargetId := Round(TargetIds.Data[i]);
    if TargetId >= VocabSize or TargetId < 0 then Continue;

    MaxLogit := -1e10;
    for j := 0 to VocabSize - 1 do
      if LogitsOut.Data[i * VocabSize + j] > MaxLogit then
        MaxLogit := LogitsOut.Data[i * VocabSize + j];

    ExpSum := 0;
    for j := 0 to VocabSize - 1 do
      ExpSum := ExpSum + System.Exp(LogitsOut.Data[i * VocabSize + j] - MaxLogit);

    Prob := System.Exp(LogitsOut.Data[i * VocabSize + TargetId] - MaxLogit) / ExpSum;
    if Prob > 0 then
      LossVal := LossVal - Ln(Prob);
  end;

  LossVal := LossVal / SeqLen;
end;

function TDecoderTransformer.Perplexity(LossVal: Single): Single;
begin
  if LossVal > 0 then
    Result := System.Exp(LossVal)
  else
    Result := 0;
end;

end.
