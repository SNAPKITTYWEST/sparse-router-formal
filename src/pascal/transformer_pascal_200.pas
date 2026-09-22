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

{ transformer_pascal_200.pas
  Pascal Transformer â€” 200 Lines
  Decoder-only transformer block: attention + MLP + layer norm + residual.
  Dependency: TensorCore.pas (TTensor, TensorMatMul, TensorAdd, etc.)
  License: AGPL-3.0-or-later
}
unit transformer_pascal_200;

interface

uses SysUtils, Math, TensorCore;

const
  ATTN_NEG_INF = -1.0e9;
  GELU_SQRT2PI = 0.7978845608028654;
  GELU_COEFF   = 0.044715;
  LN_EPS       = 1.0e-5;

type
  TAttnWeights = record
    WQ, WK, WV, WO : TTensor;   { each D_model x D_head }
    D_model, D_head, N_heads : Integer;
  end;

  TMLPWeights = record
    W1, B1, W2, B2 : TTensor;  { W1: D_ff x D_model, W2: D_model x D_ff }
    D_model, D_ff  : Integer;
  end;

  TLNWeights = record
    Gamma, Beta : TTensor;       { D_model }
    D_model     : Integer;
  end;

  TBlockWeights = record
    Attn    : TAttnWeights;
    MLP     : TMLPWeights;
    LN1, LN2: TLNWeights;
    D_model : Integer;
  end;

{ Layer normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta }
function LayerNorm(const X, Gamma, Beta: TTensor; N, D: Integer): TTensor;

{ GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) }
function GELU(const X: TTensor; Len: Integer): TTensor;

{ Causal softmax: numerically stable, masks upper triangle }
function CausalSoftmax(const Scores: TTensor; N: Integer): TTensor;

{ Scaled dot-product attention for one head }
function SingleHeadAttention(const Q, K, V: TTensor; N, DH: Integer): TTensor;

{ Full transformer block forward: X -> X + Attn(LN1(X)); Y + MLP(LN2(Y)) }
function BlockForward(const W: TBlockWeights; const X: TTensor; N: Integer): TTensor;

implementation

function LayerNorm(const X, Gamma, Beta: TTensor; N, D: Integer): TTensor;
var
  I, Dim: Integer;
  Mu, V, Inv: Double;
begin
  Result := TensorCreate([N, D]);
  for I := 0 to N - 1 do
  begin
    Mu := 0;
    for Dim := 0 to D - 1 do Mu := Mu + TensorGet(X, I * D + Dim);
    Mu := Mu / D;
    V := 0;
    for Dim := 0 to D - 1 do V := V + Sqr(TensorGet(X, I * D + Dim) - Mu);
    V := V / D;
    Inv := 1.0 / Sqrt(V + LN_EPS);
    for Dim := 0 to D - 1 do
      TensorSet(Result, I * D + Dim,
        TensorGet(Gamma, Dim) * (TensorGet(X, I * D + Dim) - Mu) * Inv + TensorGet(Beta, Dim));
  end;
end;

function GELU(const X: TTensor; Len: Integer): TTensor;
var
  I: Integer;
  XV, X3, Inner, ExpV: Double;
begin
  Result := TensorCreate([Len]);
  for I := 0 to Len - 1 do
  begin
    XV    := TensorGet(X, I);
    X3    := XV * XV * XV;
    Inner := GELU_SQRT2PI * (XV + GELU_COEFF * X3);
    ExpV  := Exp(2.0 * Inner);
    TensorSet(Result, I, 0.5 * XV * (1.0 + (ExpV - 1.0) / (ExpV + 1.0)));
  end;
end;

function CausalSoftmax(const Scores: TTensor; N: Integer): TTensor;
var
  I, J: Integer;
  MxV, SumV, V: Double;
begin
  Result := TensorCreate([N, N]);
  for I := 0 to N - 1 do
  begin
    MxV := ATTN_NEG_INF;
    for J := 0 to N - 1 do
    begin
      V := if J > I then ATTN_NEG_INF else TensorGet(Scores, I * N + J);
      if V > MxV then MxV := V;
    end;
    SumV := 0;
    for J := 0 to N - 1 do
    begin
      V := if J > I then ATTN_NEG_INF else TensorGet(Scores, I * N + J);
      TensorSet(Result, I * N + J, Exp(V - MxV));
      SumV := SumV + TensorGet(Result, I * N + J);
    end;
    if SumV > 1.0e-12 then
      for J := 0 to N - 1 do
        TensorSet(Result, I * N + J, TensorGet(Result, I * N + J) / SumV);
  end;
end;

function SingleHeadAttention(const Q, K, V: TTensor; N, DH: Integer): TTensor;
var
  I, J, Dim: Integer;
  Scale, Dot, Acc: Double;
  Scores, Weights: TTensor;
begin
  Scale  := 1.0 / Sqrt(DH);
  Scores := TensorCreate([N, N]);
  for I := 0 to N - 1 do
    for J := 0 to N - 1 do
    begin
      Dot := 0;
      for Dim := 0 to DH - 1 do
        Dot := Dot + TensorGet(Q, I * DH + Dim) * TensorGet(K, J * DH + Dim);
      TensorSet(Scores, I * N + J, Dot * Scale);
    end;
  Weights := CausalSoftmax(Scores, N);
  Result  := TensorCreate([N, DH]);
  for I := 0 to N - 1 do
    for Dim := 0 to DH - 1 do
    begin
      Acc := 0;
      for J := 0 to N - 1 do
        Acc := Acc + TensorGet(Weights, I * N + J) * TensorGet(V, J * DH + Dim);
      TensorSet(Result, I * DH + Dim, Acc);
    end;
end;

function BlockForward(const W: TBlockWeights; const X: TTensor; N: Integer): TTensor;
var
  DM, DH: Integer;
  LN1Out, Q, K, V, AttnOut, Res1, LN2Out, Pre, Act, MlpOut: TTensor;
  I, Dim, K2: Integer;
  Acc: Double;
begin
  DM := W.D_model;
  DH := W.Attn.D_head;

  { Pre-norm attention }
  LN1Out := LayerNorm(X, W.LN1.Gamma, W.LN1.Beta, N, DM);

  Q := TensorCreate([N, DH]); K := TensorCreate([N, DH]); V := TensorCreate([N, DH]);
  for I := 0 to N - 1 do
    for Dim := 0 to DH - 1 do
    begin
      Acc := 0; for K2 := 0 to DM - 1 do Acc := Acc + TensorGet(LN1Out, I*DM+K2) * TensorGet(W.Attn.WQ, Dim*DM+K2);
      TensorSet(Q, I*DH+Dim, Acc);
      Acc := 0; for K2 := 0 to DM - 1 do Acc := Acc + TensorGet(LN1Out, I*DM+K2) * TensorGet(W.Attn.WK, Dim*DM+K2);
      TensorSet(K, I*DH+Dim, Acc);
      Acc := 0; for K2 := 0 to DM - 1 do Acc := Acc + TensorGet(LN1Out, I*DM+K2) * TensorGet(W.Attn.WV, Dim*DM+K2);
      TensorSet(V, I*DH+Dim, Acc);
    end;

  AttnOut := SingleHeadAttention(Q, K, V, N, DH);

  { Project back and residual }
  Res1 := TensorCreate([N, DM]);
  for I := 0 to N - 1 do
    for Dim := 0 to DM - 1 do
    begin
      Acc := 0; for K2 := 0 to DH - 1 do Acc := Acc + TensorGet(AttnOut, I*DH+K2) * TensorGet(W.Attn.WO, Dim*DH+K2);
      TensorSet(Res1, I*DM+Dim, TensorGet(X, I*DM+Dim) + Acc);
    end;

  { Pre-norm MLP }
  LN2Out := LayerNorm(Res1, W.LN2.Gamma, W.LN2.Beta, N, DM);

  Pre := TensorCreate([N * W.MLP.D_ff]);
  for I := 0 to N - 1 do
    for Dim := 0 to W.MLP.D_ff - 1 do
    begin
      Acc := TensorGet(W.MLP.B1, Dim);
      for K2 := 0 to DM - 1 do Acc := Acc + TensorGet(LN2Out, I*DM+K2) * TensorGet(W.MLP.W1, Dim*DM+K2);
      TensorSet(Pre, I*W.MLP.D_ff+Dim, Acc);
    end;
  Act := GELU(Pre, N * W.MLP.D_ff);

  MlpOut := TensorCreate([N, DM]);
  for I := 0 to N - 1 do
    for Dim := 0 to DM - 1 do
    begin
      Acc := TensorGet(W.MLP.B2, Dim);
      for K2 := 0 to W.MLP.D_ff - 1 do Acc := Acc + TensorGet(Act, I*W.MLP.D_ff+K2) * TensorGet(W.MLP.W2, Dim*W.MLP.D_ff+K2);
      TensorSet(MlpOut, I*DM+Dim, Acc);
    end;

  { Final residual }
  Result := TensorCreate([N, DM]);
  for I := 0 to N * DM - 1 do
    TensorSet(Result, I, TensorGet(Res1, I) + TensorGet(MlpOut, I));
end;

end.
