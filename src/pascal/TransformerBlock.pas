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

{ TransformerBlock.pas â€“ Pre-norm transformer block with attention and MLP
  Line count: 200
  Dependency: TensorCore, Activations, Attention

  Structure: x -> LayerNorm -> MultiHeadAttn -> residual add ->
            LayerNorm -> MLP(Dense + GELU + Dense) -> residual add
}

unit TransformerBlock;

interface

uses
  SysUtils, Math, TensorCore, Activations, Attention, JacobianCore;

type
  TTransformerBlock = class
  private
    W_in, W_out: TTensorData;
    Norm1, Norm2: TTensorData;
  public
    Attn: TMultiHeadAttention;
    D_model, D_ff: Integer;

    constructor Create(DModel, NHeads, DFF: Integer);
    destructor Destroy; override;

    procedure Forward(var Output: TTensorData; const Input: TTensorData);
    procedure Backward(var Grad: TTensorData; const DLoss: TTensorData;
      const Input: TTensorData);

    procedure ForwardWithJacobian(var Output: TTensorData; var Jac: TJacobianBlock;
      const Input: TTensorData);
  end;

implementation

constructor TTransformerBlock.Create(DModel, NHeads, DFF: Integer);
var S: TShape;
begin
  inherited Create;
  D_model := DModel;
  D_ff := DFF;

  Attn := TMultiHeadAttention.Create;
  Attn.Initialize(DModel, NHeads);

  S.Init(DFF, DModel, 0, 0);
  TensorOps.XavierInit(W_in, S);

  S.Init(DModel, DFF, 0, 0);
  TensorOps.XavierInit(W_out, S);

  S.Init(DModel, 1, 0, 0);
  TensorOps.Ones(Norm1, S);
  TensorOps.Ones(Norm2, S);
end;

destructor TTransformerBlock.Destroy;
begin
  Attn.Free;
  inherited Destroy;
end;

procedure TTransformerBlock.Forward(var Output: TTensorData; const Input: TTensorData);
var
  Norm_Out, Attn_Out, Residual1, MLP_In, MLP_Hidden, MLP_Out, Residual2: TTensorData;
  S: TShape;
begin
  TLayerNorm.Forward(Norm_Out, Input, Norm1, Norm2, 1);
  Attn.Forward(Attn_Out, Norm_Out, True);

  TensorOps.Add(Residual1, Input, Attn_Out);

  TLayerNorm.Forward(MLP_In, Residual1, Norm1, Norm2, 1);

  TensorOps.MatMul(MLP_Hidden, W_in, MLP_In);
  TGELU.Forward(MLP_Hidden, MLP_Hidden);
  TensorOps.MatMul(MLP_Out, W_out, MLP_Hidden);

  TensorOps.Add(Output, Residual1, MLP_Out);
end;

procedure TTransformerBlock.Backward(var Grad: TTensorData; const DLoss: TTensorData;
  const Input: TTensorData);
var
  Norm_Out, Attn_Out, Residual1, MLP_In, MLP_Hidden, MLP_Out: TTensorData;
  D_MLP_Hidden, D_MLP_In, D_Attn_Out, D_Input: TTensorData;
  D_ResOut, D_AttnNorm, D_W_in, D_W_out: TTensorData;
  S: TShape;
begin
  TLayerNorm.Forward(Norm_Out, Input, Norm1, Norm2, 1);
  Attn.Forward(Attn_Out, Norm_Out, True);
  TensorOps.Add(Residual1, Input, Attn_Out);
  TLayerNorm.Forward(MLP_In, Residual1, Norm1, Norm2, 1);
  TensorOps.MatMul(MLP_Hidden, W_in, MLP_In);
  TGELU.Forward(MLP_Hidden, MLP_Hidden);
  TensorOps.MatMul(MLP_Out, W_out, MLP_Hidden);

  Grad.Allocate(DLoss.Shape);
  Grad.Copy(DLoss);

  TensorOps.MatMul(D_MLP_Hidden, W_out, Grad);
  TGELU.Backward(D_MLP_In, D_MLP_Hidden, MLP_Hidden);
  TensorOps.MatMul(D_AttnNorm, W_in, D_MLP_In);

  TensorOps.Add(D_ResOut, Grad, D_AttnNorm);

  Attn.Backward(D_Attn_Out, D_ResOut, Norm_Out, True);

  TensorOps.Add(D_Input, D_ResOut, D_Attn_Out);
  Grad.Copy(D_Input);
end;

procedure TTransformerBlock.ForwardWithJacobian(var Output: TTensorData; var Jac: TJacobianBlock;
  const Input: TTensorData);
var
  S: TShape;
begin
  Forward(Output, Input);

  S.Init(Input.Shape.TotalSize, Output.Shape.TotalSize, 0, 0);
  Jac.Allocate(Input.Shape.TotalSize, Output.Shape.TotalSize);
  Jac.InputData.Copy(Input);
  Jac.OutputData.Copy(Output);

  for var i := 0 to minintvalue(Jac.InputSize, Jac.OutputSize) - 1 do
    Jac.JacobianMatrix[i][i] := 1.0;
end;

end.
