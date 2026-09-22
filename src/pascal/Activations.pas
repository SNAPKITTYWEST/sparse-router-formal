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

{ Activations.pas â€“ Activation functions with analytic backward passes
  Line count: 150
  Dependency: TensorCore
}

unit Activations;

interface

uses
  SysUtils, Math, TensorCore;

type
  TSoftmax = class
  public
    class procedure Forward(var Output: TTensorData; const Input: TTensorData; Axis: Integer = 1);
    class procedure Backward(var Grad: TTensorData; const DLoss: TTensorData;
      const ForwardOutput: TTensorData; Axis: Integer = 1);
  end;

  TGELU = class
  public
    class procedure Forward(var Output: TTensorData; const Input: TTensorData);
    class procedure Backward(var Grad: TTensorData; const DLoss: TTensorData;
      const Input, ForwardOutput: TTensorData);
  end;

  TLayerNorm = class
  public
    class procedure Forward(var Output: TTensorData; const Input: TTensorData;
      var Mean, Var: TTensorData; Axis: Integer; Eps: Single = 1e-6);
    class procedure Backward(var Grad: TTensorData; const DLoss: TTensorData;
      const Input, Mean, Var: TTensorData; Axis: Integer; Eps: Single = 1e-6);
  end;

  TReLU = class
  public
    class procedure Forward(var Output: TTensorData; const Input: TTensorData);
    class procedure Backward(var Grad: TTensorData; const DLoss: TTensorData; const Input: TTensorData);
  end;

implementation

class procedure TSoftmax.Forward(var Output: TTensorData; const Input: TTensorData; Axis: Integer = 1);
var
  i, j, AxisSize, OutSize: Integer;
  MaxVal, Sum, Exp: Single;
begin
  Output.Allocate(Input.Shape);
  OutSize := Input.Shape.TotalSize div Input.Shape.Dims[Axis];
  AxisSize := Input.Shape.Dims[Axis];

  for i := 0 to OutSize - 1 do begin
    MaxVal := -1e10;
    for j := 0 to AxisSize - 1 do
      if Input.Data[i * AxisSize + j] > MaxVal then
        MaxVal := Input.Data[i * AxisSize + j];

    Sum := 0;
    for j := 0 to AxisSize - 1 do begin
      Exp := System.Exp(Input.Data[i * AxisSize + j] - MaxVal);
      Output.Data[i * AxisSize + j] := Exp;
      Sum := Sum + Exp;
    end;

    for j := 0 to AxisSize - 1 do
      Output.Data[i * AxisSize + j] := Output.Data[i * AxisSize + j] / Sum;
  end;
end;

class procedure TSoftmax.Backward(var Grad: TTensorData; const DLoss: TTensorData;
  const ForwardOutput: TTensorData; Axis: Integer = 1);
var
  i, j, k, AxisSize, OutSize: Integer;
  JacRow, Sum: Single;
begin
  Grad.Allocate(DLoss.Shape);
  Grad.Fill(0);
  OutSize := ForwardOutput.Shape.TotalSize div ForwardOutput.Shape.Dims[Axis];
  AxisSize := ForwardOutput.Shape.Dims[Axis];

  for i := 0 to OutSize - 1 do begin
    for j := 0 to AxisSize - 1 do begin
      Sum := 0;
      for k := 0 to AxisSize - 1 do begin
        if j = k then
          JacRow := ForwardOutput.Data[i * AxisSize + j] *
                    (1 - ForwardOutput.Data[i * AxisSize + k])
        else
          JacRow := -ForwardOutput.Data[i * AxisSize + j] *
                     ForwardOutput.Data[i * AxisSize + k];
        Sum := Sum + JacRow * DLoss.Data[i * AxisSize + k];
      end;
      Grad.Data[i * AxisSize + j] := Sum;
    end;
  end;
end;

class procedure TGELU.Forward(var Output: TTensorData; const Input: TTensorData);
var
  i: Integer;
  x, x3, TanhArg, Cdf: Single;
const
  SQRT2PI = 0.7978845608;
  COEFF = 0.044715;
begin
  Output.Allocate(Input.Shape);
  for i := 0 to Length(Input.Data) - 1 do begin
    x := Input.Data[i];
    x3 := x * x * x;
    TanhArg := SQRT2PI * (x + COEFF * x3);
    Cdf := 0.5 * (1 + System.Tanh(TanhArg));
    Output.Data[i] := x * Cdf;
  end;
end;

class procedure TGELU.Backward(var Grad: TTensorData; const DLoss: TTensorData;
  const Input, ForwardOutput: TTensorData);
var
  i: Integer;
  x, x2, x3, TanhArg, Cdf, Sech2, Pdf, Deriv: Single;
const
  SQRT2PI = 0.7978845608;
  COEFF = 0.044715;
begin
  Grad.Allocate(DLoss.Shape);
  for i := 0 to Length(Input.Data) - 1 do begin
    x := Input.Data[i];
    x2 := x * x;
    x3 := x2 * x;
    TanhArg := SQRT2PI * (x + COEFF * x3);
    Cdf := 0.5 * (1 + System.Tanh(TanhArg));
    Sech2 := 1 - Sqr(System.Tanh(TanhArg));
    Pdf := SQRT2PI * Sech2 * (1 + 3 * COEFF * x2) * 0.5;
    Deriv := Cdf + x * Pdf;
    Grad.Data[i] := DLoss.Data[i] * Deriv;
  end;
end;

class procedure TLayerNorm.Forward(var Output: TTensorData; const Input: TTensorData;
  var Mean, Var: TTensorData; Axis: Integer; Eps: Single = 1e-6);
var
  i, AxisSize, OutSize, j: Integer;
  Sum, M, V, StdDev: Single;
  MeanShape, VarShape: TShape;
begin
  if Axis >= Input.Shape.Rank then
    raise Exception.Create('LayerNorm axis out of bounds');

  AxisSize := Input.Shape.Dims[Axis];
  OutSize := Input.Shape.TotalSize div AxisSize;

  MeanShape := Input.Shape;
  MeanShape.Dims[Axis] := 1;
  Mean.Allocate(MeanShape);
  Var.Allocate(MeanShape);

  Output.Allocate(Input.Shape);

  for i := 0 to OutSize - 1 do begin
    Sum := 0;
    for j := 0 to AxisSize - 1 do
      Sum := Sum + Input.Data[i * AxisSize + j];
    M := Sum / AxisSize;
    Mean.Data[i] := M;

    Sum := 0;
    for j := 0 to AxisSize - 1 do
      Sum := Sum + Sqr(Input.Data[i * AxisSize + j] - M);
    V := Sum / AxisSize;
    Var.Data[i] := V;

    StdDev := Sqrt(V + Eps);
    for j := 0 to AxisSize - 1 do
      Output.Data[i * AxisSize + j] := (Input.Data[i * AxisSize + j] - M) / StdDev;
  end;
end;

class procedure TLayerNorm.Backward(var Grad: TTensorData; const DLoss: TTensorData;
  const Input, Mean, Var: TTensorData; Axis: Integer; Eps: Single = 1e-6);
var
  i, j, AxisSize, OutSize: Integer;
  StdDev, SumDLoss, SumDLossX, C: Single;
begin
  AxisSize := Input.Shape.Dims[Axis];
  OutSize := Input.Shape.TotalSize div AxisSize;
  Grad.Allocate(Input.Shape);

  for i := 0 to OutSize - 1 do begin
    StdDev := Sqrt(Var.Data[i] + Eps);
    SumDLoss := 0;
    SumDLossX := 0;
    for j := 0 to AxisSize - 1 do begin
      SumDLoss := SumDLoss + DLoss.Data[i * AxisSize + j];
      SumDLossX := SumDLossX + DLoss.Data[i * AxisSize + j] *
                               (Input.Data[i * AxisSize + j] - Mean.Data[i]);
    end;

    for j := 0 to AxisSize - 1 do begin
      C := (Input.Data[i * AxisSize + j] - Mean.Data[i]) / (Var.Data[i] + Eps);
      Grad.Data[i * AxisSize + j] :=
        (DLoss.Data[i * AxisSize + j] / StdDev) -
        (SumDLoss / (AxisSize * StdDev)) -
        (SumDLossX * C / (AxisSize * (Var.Data[i] + Eps)));
    end;
  end;
end;

class procedure TReLU.Forward(var Output: TTensorData; const Input: TTensorData);
var i: Integer;
begin
  Output.Allocate(Input.Shape);
  for i := 0 to Length(Input.Data) - 1 do
    if Input.Data[i] > 0 then
      Output.Data[i] := Input.Data[i]
    else
      Output.Data[i] := 0;
end;

class procedure TReLU.Backward(var Grad: TTensorData; const DLoss: TTensorData; const Input: TTensorData);
var i: Integer;
begin
  Grad.Allocate(DLoss.Shape);
  for i := 0 to Length(Input.Data) - 1 do
    if Input.Data[i] > 0 then
      Grad.Data[i] := DLoss.Data[i]
    else
      Grad.Data[i] := 0;
end;

end.
