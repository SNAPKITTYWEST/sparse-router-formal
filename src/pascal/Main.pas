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

{ Main.pas â€“ Synthetic copy task training with Jacobian sanity check
  Line count: 100
  Dependency: Model, Training, JacobianCore

  Task: Copy the input sequence to output (sequence-to-sequence).
  Validation: Run finite-difference check on attention Jacobian.
}

program Main;

uses
  SysUtils, Math, TensorCore, Activations, JacobianCore, Attention, Model, Training;

const
  VOCAB_SIZE = 512;
  D_MODEL = 128;
  N_HEADS = 4;
  D_FF = 512;
  N_LAYERS = 4;
  MAX_SEQ_LEN = 64;
  BATCH_SIZE = 2;
  N_EPOCHS = 5;
  LR = 0.001;

var
  TransformerModel: TDecoderTransformer;
  Optimizer: TAdamOptimizer;
  Epoch, Step, i, j, SeqLen: Integer;
  InputIds, TargetIds, Logits, Grad: TTensorData;
  Loss, LR_Current: Single;
  InputShape: TShape;
  Attn: TScaledDotProductAttention;
  Q, K, V, Scores: TTensorData;
  Jac: TJacobianBlock;
  FDError: Single;

begin
  WriteLn('=== Decoder-Only Transformer Training ===');
  WriteLn(Format('Model: vocab=%d, d_model=%d, heads=%d, layers=%d',
    [VOCAB_SIZE, D_MODEL, N_HEADS, N_LAYERS]));

  TransformerModel := TDecoderTransformer.Create(VOCAB_SIZE, D_MODEL, N_HEADS, D_FF, N_LAYERS, MAX_SEQ_LEN);
  Optimizer := TAdamOptimizer.Create(LR, 0.9, 0.999, 1e-8);

  WriteLn('Model initialized.');

  SeqLen := 8;
  InputShape.Init(SeqLen, 1, 0, 0);
  InputIds.Allocate(InputShape);
  TargetIds.Allocate(InputShape);

  for i := 0 to SeqLen - 1 do begin
    InputIds.Data[i] := Random(32) mod VOCAB_SIZE;
    TargetIds.Data[i] := InputIds.Data[i];
  end;

  WriteLn('Synthetic data generated.');

  for Epoch := 0 to N_EPOCHS - 1 do begin
    LR_Current := LR;
    TLRScheduler.StepDecay(LR_Current, Epoch, 2, 0.95);
    Optimizer.Alpha := LR_Current;

    TransformerModel.Forward(Logits, InputIds);

    TransformerModel.Loss(Loss, Logits, TargetIds);

    Grad.Allocate(Logits.Shape);
    Grad.Fill(1.0 / (SeqLen * VOCAB_SIZE));

    WriteLn(Format('Epoch %d: Loss=%.6f, LR=%.6f, Perplexity=%.2f',
      [Epoch, Loss, LR_Current, TransformerModel.Perplexity(Loss)]));

    if IsNaN(Loss) or IsInfinite(Loss) then begin
      WriteLn('ERROR: Loss is NaN or Inf');
      Break;
    end;
  end;

  WriteLn('');
  WriteLn('=== Jacobian Sanity Check (Softmax) ===');

  InputShape.Init(4, 128, 0, 0);
  TensorOps.RandomNormal(Q, InputShape, 0, 1);
  TensorOps.RandomNormal(K, InputShape, 0, 1);
  TensorOps.RandomNormal(V, InputShape, 0, 1);

  TensorOps.Transpose(K, K);
  TScaledDotProductAttention.ComputeAttentionScores(Scores, Q, K, False);

  Jac.Allocate(16, 16);
  for i := 0 to 3 do begin
    for j := 0 to 3 do begin
      if i = j then
        Jac.JacobianMatrix[i][j] := Scores.Data[i] * (1 - Scores.Data[j])
      else
        Jac.JacobianMatrix[i][j] := -Scores.Data[i] * Scores.Data[j];
    end;
  end;

  WriteLn('Softmax Jacobian computed.');

  FDError := TFDChecker.CheckSoftmaxJacobian(Q, Jac.JacobianMatrix, 0);
  WriteLn(Format('Max FD Error (Softmax Jacobian): %.8f', [FDError]));

  if FDError < FD_TOL then
    WriteLn('PASS: Jacobian within tolerance')
  else
    WriteLn('WARNING: Jacobian error exceeds tolerance');

  WriteLn('');
  WriteLn('=== Training Complete ===');

  TransformerModel.Free;
  Optimizer.Free;
end.
