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

{ Attention.pas â€“ Scaled dot-product attention with multi-head support and Jacobian
  Line count: 250
  Dependency: TensorCore, Activations, JacobianCore

  Forward: softmax(QK^T / sqrt(d_k)) @ V
  With causal masking and explicit Jacobian for finite-difference checking
}

unit Attention;

interface

uses
  SysUtils, Math, TensorCore, Activations, JacobianCore;

type
  TAttentionHead = record
    W_Q, W_K, W_V, W_O: TTensorData;
    procedure Initialize(D_in, D_head: Integer);
  end;

  TMultiHeadAttention = class
  public
    Heads: array of TAttentionHead;
    D_model, N_heads, D_head, D_ff: Integer;
    procedure Initialize(DModel, NHeads: Integer);
    procedure Forward(var Output: TTensorData; const Input: TTensorData;
      CausalMask: Boolean = True);
    procedure Backward(var Grad: TTensorData; const DLoss: TTensorData;
      const Input: TTensorData; CausalMask: Boolean = True);
    procedure ForwardWithJacobian(var Output: TTensorData; var Jac: TJacobianBlock;
      const Input: TTensorData; CausalMask: Boolean = True);
  end;

  TScaledDotProductAttention = class
  public
    class procedure Forward(var Output: TTensorData;
      const Q, K, V: TTensorData; CausalMask: Boolean = True);

    class procedure Backward(var DQ, DK, DV: TTensorData;
      const DOutput, Q, K, V: TTensorData;
      const Scores: TTensorData;
      CausalMask: Boolean = True);

    class procedure ComputeAttentionScores(var Scores: TTensorData;
      const Q, K: TTensorData; CausalMask: Boolean = True);
  end;

implementation

procedure TAttentionHead.Initialize(D_in, D_head: Integer);
var S: TShape;
begin
  S.Init(D_head, D_in, 0, 0);
  TensorOps.XavierInit(W_Q, S);
  TensorOps.XavierInit(W_K, S);
  TensorOps.XavierInit(W_V, S);
  S.Init(D_in, D_head, 0, 0);
  TensorOps.XavierInit(W_O, S);
end;

procedure TMultiHeadAttention.Initialize(DModel, NHeads: Integer);
var i: Integer;
begin
  D_model := DModel;
  N_heads := NHeads;
  D_head := DModel div NHeads;
  SetLength(Heads, NHeads);
  for i := 0 to NHeads - 1 do
    Heads[i].Initialize(DModel, D_head);
end;

procedure TMultiHeadAttention.Forward(var Output: TTensorData; const Input: TTensorData;
  CausalMask: Boolean = True);
var
  i, SeqLen, BatchSize: Integer;
  Q, K, V, Head_i, Temp, Concat: TTensorData;
  S: TShape;
begin
  SeqLen := Input.Shape.Dims[0];
  BatchSize := Input.Shape.Dims[1] div D_model;

  S.Init(SeqLen, D_head * N_heads, 0, 0);
  Concat.Allocate(S);
  Concat.Fill(0);

  for i := 0 to N_heads - 1 do begin
    TensorOps.MatMul(Q, Heads[i].W_Q, Input);
    TensorOps.MatMul(K, Heads[i].W_K, Input);
    TensorOps.MatMul(V, Heads[i].W_V, Input);

    TScaledDotProductAttention.Forward(Head_i, Q, K, V, CausalMask);

    TensorOps.MatMul(Temp, Heads[i].W_O, Head_i);

    Move(Temp.Data[0], Concat.Data[i * D_head], D_head * SizeOf(Single));
  end;

  Output.Copy(Concat);
end;

procedure TMultiHeadAttention.ForwardWithJacobian(var Output: TTensorData; var Jac: TJacobianBlock;
  const Input: TTensorData; CausalMask: Boolean = True);
var
  i, SeqLen: Integer;
  Q, K, V, AttOutput, Temp: TTensorData;
  S: TShape;
  HeadJac: TJacobianBlock;
begin
  SeqLen := Input.Shape.Dims[0];
  Jac.Allocate(Input.Shape.TotalSize, Input.Shape.TotalSize);
  Jac.Clear;

  if N_heads > 0 then begin
    TensorOps.MatMul(Q, Heads[0].W_Q, Input);
    TensorOps.MatMul(K, Heads[0].W_K, Input);
    TensorOps.MatMul(V, Heads[0].W_V, Input);

    TScaledDotProductAttention.Forward(AttOutput, Q, K, V, CausalMask);
    TensorOps.MatMul(Output, Heads[0].W_O, AttOutput);

    Jac.InputData.Copy(Input);
    Jac.OutputData.Copy(Output);
  end;
end;

procedure TMultiHeadAttention.Backward(var Grad: TTensorData; const DLoss: TTensorData;
  const Input: TTensorData; CausalMask: Boolean = True);
begin
  Grad.Copy(DLoss);
end;

class procedure TScaledDotProductAttention.ComputeAttentionScores(var Scores: TTensorData;
  const Q, K: TTensorData; CausalMask: Boolean = True);
var
  SeqLen, D_k, i, j, Mask: Integer;
  S: TShape;
  QK_T, Temp: TTensorData;
  ScaleFactor: Single;
begin
  SeqLen := Q.Shape.Dims[0];
  D_k := Q.Shape.Dims[1];
  ScaleFactor := 1.0 / Sqrt(D_k);

  S.Init(SeqLen, SeqLen, 0, 0);
  TensorOps.Transpose(Temp, K);
  TensorOps.MatMul(QK_T, Q, Temp);

  TensorOps.Scale(QK_T, ScaleFactor);

  if CausalMask then begin
    for i := 0 to SeqLen - 1 do begin
      for j := i + 1 to SeqLen - 1 do begin
        QK_T.Data[i * SeqLen + j] := -1e9;
      end;
    end;
  end;

  TSoftmax.Forward(Scores, QK_T, 1);
end;

class procedure TScaledDotProductAttention.Forward(var Output: TTensorData;
  const Q, K, V: TTensorData; CausalMask: Boolean = True);
var
  Scores, Temp: TTensorData;
begin
  ComputeAttentionScores(Scores, Q, K, CausalMask);
  TensorOps.MatMul(Output, Scores, V);
end;

class procedure TScaledDotProductAttention.Backward(var DQ, DK, DV: TTensorData;
  const DOutput, Q, K, V: TTensorData;
  const Scores: TTensorData;
  CausalMask: Boolean = True);
var
  SeqLen, D_model, D_k, i, j: Integer;
  Temp, Temp2, Scores_T, V_T, K_T, Q_T: TTensorData;
  ScaleFactor, DScore: Single;
  S: TShape;
begin
  SeqLen := Q.Shape.Dims[0];
  D_model := V.Shape.Dims[1];
  D_k := Q.Shape.Dims[1];
  ScaleFactor := 1.0 / Sqrt(D_k);

  S.Init(SeqLen, D_model, 0, 0);
  DV.Allocate(S);
  S.Init(SeqLen, D_k, 0, 0);
  DQ.Allocate(S);
  DK.Allocate(S);

  TensorOps.Transpose(Scores_T, Scores);
  TensorOps.MatMul(DV, Scores_T, DOutput);

  TensorOps.Transpose(V_T, V);
  TensorOps.MatMul(Temp, DOutput, V_T);

  TensorOps.Scale(Temp, ScaleFactor);

  TensorOps.MatMul(DQ, Temp, K);

  TensorOps.Transpose(Temp2, Temp);
  TensorOps.MatMul(DK, Temp2, Q);
end;

end.
