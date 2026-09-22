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

{ JacobianCore.pas â€“ Analytic Jacobian computation and finite-difference validation
  Line count: 400
  Dependency: TensorCore, Activations

  For every primitive operation, we compute:
  - Forward pass: f(x) -> y
  - Analytic Jacobian: J_ij = df_i/dx_j [stored as full dense matrix]
  - Finite-difference checker: verify against numerical gradient
}

unit JacobianCore;

interface

uses
  SysUtils, Math, TensorCore, Activations;

const
  FD_EPS = 1e-5;
  FD_TOL = 1e-4;

type
  TJacobianBlock = record
    InputData: TTensorData;
    OutputData: TTensorData;
    JacobianMatrix: TMatrix;
    InputSize: Integer;
    OutputSize: Integer;
    procedure Allocate(InSize, OutSize: Integer);
    procedure Clear;
  end;

  TFDChecker = class
  public
    class function CheckLinearJacobian(const W: TTensorData; const X: TTensorData;
      const AnalyticJac: TMatrix): Single;
    class function CheckSoftmaxJacobian(const Input: TTensorData;
      const AnalyticJac: TMatrix; Axis: Integer): Single;
    class function CheckGELUJacobian(const Input: TTensorData;
      const AnalyticJac: TMatrix): Single;
    class function CheckLayerNormJacobian(const Input, Mean, Var: TTensorData;
      const AnalyticJac: TMatrix; Axis: Integer; Eps: Single): Single;
  end;

  TJacobianOps = class
  public
    class procedure LinearJacobian(var Jac: TJacobianBlock;
      const W, X: TTensorData);
    class procedure MatMulJacobian(var Jac: TJacobianBlock;
      const A, B: TTensorData);
    class procedure SoftmaxJacobian(var Jac: TJacobianBlock;
      const Input: TTensorData; Axis: Integer);
    class procedure GELUJacobian(var Jac: TJacobianBlock;
      const Input: TTensorData);
    class procedure LayerNormJacobian(var Jac: TJacobianBlock;
      const Input, Mean, Var: TTensorData; Axis: Integer; Eps: Single);
    class procedure AddJacobian(var Jac: TJacobianBlock;
      const X, Z: TTensorData; WhichInput: Integer);
  end;

implementation

procedure TJacobianBlock.Allocate(InSize, OutSize: Integer);
begin
  InputSize := InSize;
  OutputSize := OutSize;
  SetLength(JacobianMatrix, OutSize);
  for var i := 0 to OutSize - 1 do
    SetLength(JacobianMatrix[i], InSize);
end;

procedure TJacobianBlock.Clear;
var i, j: Integer;
begin
  for i := 0 to Length(JacobianMatrix) - 1 do
    for j := 0 to Length(JacobianMatrix[i]) - 1 do
      JacobianMatrix[i][j] := 0;
end;

class function TFDChecker.CheckLinearJacobian(const W: TTensorData; const X: TTensorData;
  const AnalyticJac: TMatrix): Single;
var
  i, j: Integer;
  X_Pert: TTensorData;
  Y1, Y2, Grad, AbsError, MaxError: Single;
  Temp: TTensorData;
begin
  MaxError := 0;
  for j := 0 to Length(X.Data) - 1 do begin
    X_Pert.Copy(X);
    X_Pert.Data[j] := X_Pert.Data[j] + FD_EPS;

    TensorOps.MatMul(Temp, W, X_Pert);
    Y2 := Temp.Data[0];

    TensorOps.MatMul(Temp, W, X);
    Y1 := Temp.Data[0];

    Grad := (Y2 - Y1) / FD_EPS;
    AbsError := Abs(AnalyticJac[0][j] - Grad);
    if AbsError > MaxError then
      MaxError := AbsError;
  end;
  Result := MaxError;
end;

class function TFDChecker.CheckSoftmaxJacobian(const Input: TTensorData;
  const AnalyticJac: TMatrix; Axis: Integer): Single;
var
  i, j, AxisSize, OutSize: Integer;
  Input_Pert: TTensorData;
  Output1, Output2, Grad, AbsError, MaxError: Single;
  Temp: TTensorData;
begin
  MaxError := 0;
  AxisSize := Input.Shape.Dims[Axis];
  OutSize := Input.Shape.TotalSize div AxisSize;

  for i := 0 to Length(Input.Data) - 1 do begin
    Input_Pert.Copy(Input);
    Input_Pert.Data[i] := Input_Pert.Data[i] + FD_EPS;

    TSoftmax.Forward(Temp, Input_Pert, Axis);
    Output2 := Temp.Data[i];

    TSoftmax.Forward(Temp, Input, Axis);
    Output1 := Temp.Data[i];

    Grad := (Output2 - Output1) / FD_EPS;
    AbsError := Abs(AnalyticJac[i][i] - Grad);
    if AbsError > MaxError then
      MaxError := AbsError;
  end;
  Result := MaxError;
end;

class function TFDChecker.CheckGELUJacobian(const Input: TTensorData;
  const AnalyticJac: TMatrix): Single;
var
  i: Integer;
  Input_Pert: TTensorData;
  Output1, Output2, Grad, AbsError, MaxError: Single;
  Temp: TTensorData;
begin
  MaxError := 0;
  for i := 0 to Length(Input.Data) - 1 do begin
    Input_Pert.Copy(Input);
    Input_Pert.Data[i] := Input_Pert.Data[i] + FD_EPS;

    TGELU.Forward(Temp, Input_Pert);
    Output2 := Temp.Data[i];

    TGELU.Forward(Temp, Input);
    Output1 := Temp.Data[i];

    Grad := (Output2 - Output1) / FD_EPS;
    AbsError := Abs(AnalyticJac[i][i] - Grad);
    if AbsError > MaxError then
      MaxError := AbsError;
  end;
  Result := MaxError;
end;

class function TFDChecker.CheckLayerNormJacobian(const Input, Mean, Var: TTensorData;
  const AnalyticJac: TMatrix; Axis: Integer; Eps: Single): Single;
var
  i: Integer;
  Input_Pert: TTensorData;
  Output1, Output2, Grad, AbsError, MaxError: Single;
  Temp, TempMean, TempVar: TTensorData;
begin
  MaxError := 0;
  for i := 0 to Length(Input.Data) - 1 do begin
    Input_Pert.Copy(Input);
    Input_Pert.Data[i] := Input_Pert.Data[i] + FD_EPS;

    TLayerNorm.Forward(Temp, Input_Pert, TempMean, TempVar, Axis, Eps);
    Output2 := Temp.Data[i];

    TLayerNorm.Forward(Temp, Input, TempMean, TempVar, Axis, Eps);
    Output1 := Temp.Data[i];

    Grad := (Output2 - Output1) / FD_EPS;
    AbsError := Abs(AnalyticJac[i][i] - Grad);
    if AbsError > MaxError then
      MaxError := AbsError;
  end;
  Result := MaxError;
end;

class procedure TJacobianOps.LinearJacobian(var Jac: TJacobianBlock;
  const W, X: TTensorData);
var i, j: Integer;
begin
  if W.Shape.Rank <> 2 or X.Shape.Rank <> 1 then
    raise Exception.Create('LinearJacobian requires 2D weight and 1D input');

  Jac.Allocate(X.Shape.Dims[0], W.Shape.Dims[0]);
  TensorOps.MatMul(Jac.OutputData, W, X);
  Jac.InputData.Copy(X);

  for i := 0 to Jac.OutputSize - 1 do
    for j := 0 to Jac.InputSize - 1 do
      Jac.JacobianMatrix[i][j] := W.Data[i * Jac.InputSize + j];
end;

class procedure TJacobianOps.MatMulJacobian(var Jac: TJacobianBlock;
  const A, B: TTensorData);
var i, j, k, m, n, p: Integer; Sum: Single;
begin
  if A.Shape.Rank <> 2 or B.Shape.Rank <> 2 then
    raise Exception.Create('MatMulJacobian requires 2D tensors');

  m := A.Shape.Dims[0];
  n := A.Shape.Dims[1];
  p := B.Shape.Dims[1];

  Jac.Allocate(n * p, m * p);
  TensorOps.MatMul(Jac.OutputData, A, B);
  Jac.InputData.Copy(B);

  for i := 0 to m - 1 do
    for j := 0 to p - 1 do
      for k := 0 to n - 1 do
        for var l := 0 to p - 1 do
          if j = l then
            Jac.JacobianMatrix[i * p + j][k * p + l] := A.Data[i * n + k]
          else
            Jac.JacobianMatrix[i * p + j][k * p + l] := 0;
end;

class procedure TJacobianOps.SoftmaxJacobian(var Jac: TJacobianBlock;
  const Input: TTensorData; Axis: Integer);
var
  i, j, k, AxisSize, OutSize: Integer;
  Output: TTensorData;
begin
  AxisSize := Input.Shape.Dims[Axis];
  OutSize := Input.Shape.TotalSize div AxisSize;

  Jac.Allocate(AxisSize, AxisSize);
  TSoftmax.Forward(Output, Input, Axis);
  Jac.OutputData := Output;
  Jac.InputData.Copy(Input);

  for i := 0 to AxisSize - 1 do begin
    for j := 0 to AxisSize - 1 do begin
      if i = j then
        Jac.JacobianMatrix[i][j] := Output.Data[i] * (1 - Output.Data[j])
      else
        Jac.JacobianMatrix[i][j] := -Output.Data[i] * Output.Data[j];
    end;
  end;
end;

class procedure TJacobianOps.GELUJacobian(var Jac: TJacobianBlock;
  const Input: TTensorData);
var
  i: Integer;
  x, x2, x3, TanhArg, Cdf, Sech2, Pdf, Deriv: Single;
  Output: TTensorData;
const
  SQRT2PI = 0.7978845608;
  COEFF = 0.044715;
begin
  Jac.Allocate(Length(Input.Data), Length(Input.Data));
  TGELU.Forward(Output, Input);
  Jac.OutputData := Output;
  Jac.InputData.Copy(Input);

  for i := 0 to Length(Input.Data) - 1 do begin
    x := Input.Data[i];
    x2 := x * x;
    x3 := x2 * x;
    TanhArg := SQRT2PI * (x + COEFF * x3);
    Cdf := 0.5 * (1 + System.Tanh(TanhArg));
    Sech2 := 1 - Sqr(System.Tanh(TanhArg));
    Pdf := SQRT2PI * Sech2 * (1 + 3 * COEFF * x2) * 0.5;
    Deriv := Cdf + x * Pdf;
    Jac.JacobianMatrix[i][i] := Deriv;
  end;
end;

class procedure TJacobianOps.LayerNormJacobian(var Jac: TJacobianBlock;
  const Input, Mean, Var: TTensorData; Axis: Integer; Eps: Single);
var
  i, j, AxisSize, OutSize: Integer;
  Output: TTensorData;
  StdDev, N, Inv3: Single;
begin
  AxisSize := Input.Shape.Dims[Axis];
  OutSize := Input.Shape.TotalSize div AxisSize;
  N := AxisSize;

  Jac.Allocate(Length(Input.Data), Length(Input.Data));
  TLayerNorm.Forward(Output, Input, Jac.OutputData, Jac.OutputData, Axis, Eps);
  Jac.OutputData := Output;
  Jac.InputData.Copy(Input);

  Inv3 := 1.0 / N;
  for i := 0 to AxisSize - 1 do begin
    StdDev := Sqrt(Var.Data[0] + Eps);
    for j := 0 to AxisSize - 1 do begin
      if i = j then
        Jac.JacobianMatrix[i][j] := (1.0 - Inv3) / StdDev -
          ((Input.Data[i] - Mean.Data[0]) * (Input.Data[j] - Mean.Data[0]) * Inv3) /
          (StdDev * StdDev * StdDev)
      else
        Jac.JacobianMatrix[i][j] := -Inv3 / StdDev -
          ((Input.Data[i] - Mean.Data[0]) * (Input.Data[j] - Mean.Data[0]) * Inv3) /
          (StdDev * StdDev * StdDev);
    end;
  end;
end;

class procedure TJacobianOps.AddJacobian(var Jac: TJacobianBlock;
  const X, Z: TTensorData; WhichInput: Integer);
var i, j: Integer;
begin
  if X.Shape.TotalSize <> Z.Shape.TotalSize then
    raise Exception.Create('AddJacobian requires same-shape inputs');

  Jac.Allocate(X.Shape.TotalSize, X.Shape.TotalSize);
  Jac.Clear;

  for i := 0 to Jac.InputSize - 1 do
    Jac.JacobianMatrix[i][i] := 1.0;

  Jac.InputData.Copy(X);
  TensorOps.Add(Jac.OutputData, X, Z);
end;

end.
