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

unit TensorCore;

{$mode objfpc}{$H+}

interface

uses
  SysUtils,
  Math;

const
  TENSORCORE_LINE_BUDGET = 250;

type
  TFloatArray = array of Double;
  TIntArray = array of Integer;

  { Explicit tensor record.
    Data is row-major and Shape contains each dimension. }
  TTensor = record
    Data: TFloatArray;
    Shape: TIntArray;
  end;

function TensorCreate(const Shape: array of Integer): TTensor;
function TensorScalar(const Value: Double): TTensor;
function TensorVector(const Values: array of Double): TTensor;
function TensorMatrix(const Rows, Cols: Integer): TTensor;

function TensorRank(const T: TTensor): Integer;
function TensorSize(const T: TTensor): Integer;
function TensorDim(const T: TTensor; const Axis: Integer): Integer;

procedure TensorFill(var T: TTensor; const Value: Double);
procedure TensorZeros(var T: TTensor);
procedure TensorOnes(var T: TTensor);

function TensorClone(const T: TTensor): TTensor;
function TensorGet(const T: TTensor; const Index: Integer): Double;
procedure TensorSet(var T: TTensor; const Index: Integer; const Value: Double);

function TensorOffset2D(
  const T: TTensor;
  const Row, Col: Integer
): Integer;

function TensorOffset3D(
  const T: TTensor;
  const A, B, C: Integer
): Integer;

function TensorGet2D(
  const T: TTensor;
  const Row, Col: Integer
): Double;

procedure TensorSet2D(
  var T: TTensor;
  const Row, Col: Integer;
  const Value: Double
);

function TensorGet3D(
  const T: TTensor;
  const A, B, C: Integer
): Double;

procedure TensorSet3D(
  var T: TTensor;
  const A, B, C: Integer;
  const Value: Double
);

function TensorAdd(
  const A, B: TTensor
): TTensor;

function TensorSub(
  const A, B: TTensor
): TTensor;

function TensorScale(
  const A: TTensor;
  const S: Double
): TTensor;

function TensorHadamard(
  const A, B: TTensor
): TTensor;

function TensorAddInPlace(
  var A: TTensor;
  const B: TTensor
): Boolean;

function TensorScaleInPlace(
  var A: TTensor;
  const S: Double
): Boolean;

function TensorMatMul(
  const A, B: TTensor
): TTensor;

function TensorTranspose2D(
  const A: TTensor
): TTensor;

function TensorTranspose(
  const A: TTensor;
  const Axis0, Axis1: Integer
): TTensor;

function TensorReshape(
  const A: TTensor;
  const NewShape: array of Integer
): TTensor;

function TensorFlatten(
  const A: TTensor
): TTensor;

function TensorDot(
  const A, B: TTensor
): Double;

function TensorSum(
  const A: TTensor
): Double;

function TensorMean(
  const A: TTensor
): Double;

function TensorMax(
  const A: TTensor
): Double;

function TensorArgMax(
  const A: TTensor
): Integer;

procedure TensorRandomSeed(
  const Seed: Integer
);

function TensorRandomUniform(
  const Shape: array of Integer;
  const MinValue, MaxValue: Double
): TTensor;

function TensorRandomNormal(
  const Shape: array of Integer;
  const Mean, StdDev: Double
): TTensor;

function TensorXavierUniform(
  const Rows, Cols: Integer
): TTensor;

function TensorXavierNormal(
  const Rows, Cols: Integer
): TTensor;

function TensorIsFinite(
  const A: TTensor
): Boolean;

function TensorMaxAbs(
  const A: TTensor
): Double;

function TensorL2Norm(
  const A: TTensor
): Double;

function TensorSameShape(
  const A, B: TTensor
): Boolean;

function TensorShapeString(
  const A: TTensor
): string;

procedure TensorSave(
  const FileName: string;
  const A: TTensor
);

function TensorLoad(
  const FileName: string
): TTensor;

procedure TensorAssert(
  const Condition: Boolean;
  const Message: string
);

implementation

function ProductOfShape(
  const Shape: array of Integer
): Integer;
var
  I: Integer;
begin
  Result := 1;
  for I := Low(Shape) to High(Shape) do
  begin
    if Shape[I] <= 0 then
      raise EArgumentException.Create(
        'Tensor dimensions must be positive'
      );
    Result := Result * Shape[I];
  end;
end;

function ProductOfShapeArray(
  const Shape: TIntArray
): Integer;
var
  I: Integer;
begin
  Result := 1;
  for I := Low(Shape) to High(Shape) do
  begin
    if Shape[I] <= 0 then
      raise EArgumentException.Create(
        'Tensor dimensions must be positive'
      );
    Result := Result * Shape[I];
  end;
end;

procedure CheckSameShape(
  const A, B: TTensor
);
begin
  if not TensorSameShape(A, B) then
    raise EArgumentException.Create(
      'Tensor shape mismatch: ' +
      TensorShapeString(A) +
      ' versus ' +
      TensorShapeString(B)
    );
end;

procedure Check2D(
  const T: TTensor;
  const Name: string
);
begin
  if TensorRank(T) <> 2 then
    raise EArgumentException.Create(
      Name + ' must be rank 2'
    );
end;

function TensorCreate(
  const Shape: array of Integer
): TTensor;
var
  I: Integer;
begin
  SetLength(Result.Shape, Length(Shape));

  for I := Low(Shape) to High(Shape) do
    Result.Shape[I] := Shape[I];

  SetLength(
    Result.Data,
    ProductOfShape(Shape)
  );

  TensorZeros(Result);
end;

function TensorScalar(
  const Value: Double
): TTensor;
begin
  Result := TensorCreate([1]);
  Result.Data[0] := Value;
end;

function TensorVector(
  const Values: array of Double
): TTensor;
var
  I: Integer;
begin
  Result := TensorCreate([Length(Values)]);

  for I := Low(Values) to High(Values) do
    Result.Data[I] := Values[I];
end;

function TensorMatrix(
  const Rows, Cols: Integer
): TTensor;
begin
  Result := TensorCreate([Rows, Cols]);
end;

function TensorRank(
  const T: TTensor
): Integer;
begin
  Result := Length(T.Shape);
end;

function TensorSize(
  const T: TTensor
): Integer;
begin
  Result := Length(T.Data);
end;

function TensorDim(
  const T: TTensor;
  const Axis: Integer
): Integer;
begin
  if (Axis < 0) or (Axis >= TensorRank(T)) then
    raise EArgumentOutOfRangeException.Create(
      'Tensor axis out of range'
    );

  Result := T.Shape[Axis];
end;

procedure TensorFill(
  var T: TTensor;
  const Value: Double
);
var
  I: Integer;
begin
  for I := 0 to High(T.Data) do
    T.Data[I] := Value;
end;

procedure TensorZeros(
  var T: TTensor
);
begin
  TensorFill(T, 0.0);
end;

procedure TensorOnes(
  var T: TTensor
);
begin
  TensorFill(T, 1.0);
end;

function TensorClone(
  const T: TTensor
): TTensor;
begin
  Result.Shape := Copy(
    T.Shape,
    0,
    Length(T.Shape)
  );

  Result.Data := Copy(
    T.Data,
    0,
    Length(T.Data)
  );
end;

function TensorGet(
  const T: TTensor;
  const Index: Integer
): Double;
begin
  if (Index < 0) or
     (Index >= Length(T.Data)) then
    raise EArgumentOutOfRangeException.Create(
      'Tensor data index out of range'
    );

  Result := T.Data[Index];
end;

procedure TensorSet(
  var T: TTensor;
  const Index: Integer;
  const Value: Double
);
begin
  if (Index < 0) or
     (Index >= Length(T.Data)) then
    raise EArgumentOutOfRangeException.Create(
      'Tensor data index out of range'
    );

  T.Data[Index] := Value;
end;

function TensorOffset2D(
  const T: TTensor;
  const Row, Col: Integer
): Integer;
begin
  Check2D(T, 'Tensor');

  if (Row < 0) or
     (Row >= T.Shape[0]) or
     (Col < 0) or
     (Col >= T.Shape[1]) then
    raise EArgumentOutOfRangeException.Create(
      '2D tensor index out of range'
    );

  Result :=
    Row * T.Shape[1] +
    Col;
end;

function TensorOffset3D(
  const T: TTensor;
  const A, B, C: Integer
): Integer;
begin
  if TensorRank(T) <> 3 then
    raise EArgumentException.Create(
      'Tensor must be rank 3'
    );

  if (A < 0) or
     (A >= T.Shape[0]) or
     (B < 0) or
     (B >= T.Shape[1]) or
     (C < 0) or
     (C >= T.Shape[2]) then
    raise EArgumentOutOfRangeException.Create(
      '3D tensor index out of range'
    );

  Result :=
    (A * T.Shape[1] * T.Shape[2]) +
    (B * T.Shape[2]) +
    C;
end;

function TensorGet2D(
  const T: TTensor;
  const Row, Col: Integer
): Double;
begin
  Result := T.Data[
    TensorOffset2D(T, Row, Col)
  ];
end;

procedure TensorSet2D(
  var T: TTensor;
  const Row, Col: Integer;
  const Value: Double
);
begin
  T.Data[
    TensorOffset2D(T, Row, Col)
  ] := Value;
end;

function TensorGet3D(
  const T: TTensor;
  const A, B, C: Integer
): Double;
begin
  Result := T.Data[
    TensorOffset3D(T, A, B, C)
  ];
end;

procedure TensorSet3D(
  var T: TTensor;
  const A, B, C: Integer;
  const Value: Double
);
begin
  T.Data[
    TensorOffset3D(T, A, B, C)
  ] := Value;
end;

function TensorAdd(
  const A, B: TTensor
): TTensor;
var
  I: Integer;
begin
  CheckSameShape(A, B);
  Result := TensorClone(A);

  for I := 0 to High(Result.Data) do
    Result.Data[I] :=
      A.Data[I] +
      B.Data[I];
end;

function TensorSub(
  const A, B: TTensor
): TTensor;
var
  I: Integer;
begin
  CheckSameShape(A, B);
  Result := TensorClone(A);

  for I := 0 to High(Result.Data) do
    Result.Data[I] :=
      A.Data[I] -
      B.Data[I];
end;

function TensorScale(
  const A: TTensor;
  const S: Double
): TTensor;
var
  I: Integer;
begin
  Result := TensorClone(A);

  for I := 0 to High(Result.Data) do
    Result.Data[I] :=
      A.Data[I] * S;
end;

function TensorHadamard(
  const A, B: TTensor
): TTensor;
var
  I: Integer;
begin
  CheckSameShape(A, B);
  Result := TensorClone(A);

  for I := 0 to High(Result.Data) do
    Result.Data[I] :=
      A.Data[I] *
      B.Data[I];
end;

function TensorAddInPlace(
  var A: TTensor;
  const B: TTensor
): Boolean;
var
  I: Integer;
begin
  Result := False;

  if not TensorSameShape(A, B) then
    Exit;

  for I := 0 to High(A.Data) do
    A.Data[I] :=
      A.Data[I] +
      B.Data[I];

  Result := True;
end;

function TensorScaleInPlace(
  var A: TTensor;
  const S: Double
): Boolean;
var
  I: Integer;
begin
  for I := 0 to High(A.Data) do
    A.Data[I] :=
      A.Data[I] * S;

  Result := True;
end;

function TensorMatMul(
  const A, B: TTensor
): TTensor;
var
  M, K, N: Integer;
  I, J, P: Integer;
  Sum: Double;
begin
  Check2D(A, 'A');
  Check2D(B, 'B');

  M := A.Shape[0];
  K := A.Shape[1];

  if B.Shape[0] <> K then
    raise EArgumentException.Create(
      'MatMul inner dimensions do not match'
    );

  N := B.Shape[1];

  Result := TensorMatrix(M, N);

  for I := 0 to M - 1 do
    for J := 0 to N - 1 do
    begin
      Sum := 0.0;

      for P := 0 to K - 1 do
        Sum := Sum +
          A.Data[I * K + P] *
          B.Data[P * N + J];

      Result.Data[I * N + J] :=
        Sum;
    end;
end;

function TensorTranspose2D(
  const A: TTensor
): TTensor;
var
  I, J: Integer;
begin
  Check2D(A, 'Tensor');

  Result :=
    TensorMatrix(
      A.Shape[1],
      A.Shape[0]
    );

  for I := 0 to A.Shape[0] - 1 do
    for J := 0 to A.Shape[1] - 1 do
      Result.Data[
        J * A.Shape[0] + I
      ] :=
        A.Data[
          I * A.Shape[1] + J
        ];
end;

function TensorTranspose(
  const A: TTensor;
  const Axis0, Axis1: Integer
): TTensor;
var
  R, I, J, K: Integer;
  NewShape: TIntArray;
  SourceIndex, TargetIndex: Integer;
  StrideSource, StrideTarget: TIntArray;
begin
  R := TensorRank(A);

  if (Axis0 < 0) or
     (Axis0 >= R) or
     (Axis1 < 0) or
     (Axis1 >= R) then
    raise EArgumentOutOfRangeException.Create(
      'Transpose axis out of range'
    );

  NewShape :=
    Copy(A.Shape, 0, R);

  NewShape[Axis0] := A.Shape[Axis1];
  NewShape[Axis1] := A.Shape[Axis0];

  Result := TensorCreate(NewShape);

  SetLength(
    StrideSource,
    R
  );

  SetLength(
    StrideTarget,
    R
  );

  StrideSource[R - 1] := 1;
  StrideTarget[R - 1] := 1;

  for I := R - 2 downto 0 do
  begin
    StrideSource[I] :=
      StrideSource[I + 1] *
      A.Shape[I + 1];

    StrideTarget[I] :=
      StrideTarget[I + 1] *
      Result.Shape[I + 1];
  end;

  for I := 0 to TensorSize(A) - 1 do
  begin
    SourceIndex := I;
    TargetIndex := 0;

    for J := 0 to R - 1 do
    begin
      K := SourceIndex div StrideSource[J];
      SourceIndex :=
        SourceIndex mod StrideSource[J];

      if J = Axis0 then
        TargetIndex :=
          TargetIndex +
          K * StrideTarget[Axis1]
      else if J = Axis1 then
        TargetIndex :=
          TargetIndex +
          K * StrideTarget[Axis0]
      else
        TargetIndex :=
          TargetIndex +
          K * StrideTarget[J];
    end;

    Result.Data[TargetIndex] :=
      A.Data[I];
  end;
end;

function TensorReshape(
  const A: TTensor;
  const NewShape: array of Integer
): TTensor;
var
  NewSize: Integer;
begin
  NewSize :=
    ProductOfShape(NewShape);

  if NewSize <> TensorSize(A) then
    raise EArgumentException.Create(
      'Reshape changes tensor element count'
    );

  Result.Shape :=
    Copy(
      NewShape,
      0,
      Length(NewShape)
    );

  Result.Data :=
    Copy(
      A.Data,
      0,
      Length(A.Data)
    );
end;

function TensorFlatten(
  const A: TTensor
): TTensor;
begin
  Result :=
    TensorReshape(
      A,
      [TensorSize(A)]
    );
end;

function TensorDot(
  const A, B: TTensor
): Double;
var
  I: Integer;
begin
  CheckSameShape(A, B);

  Result := 0.0;

  for I := 0 to High(A.Data) do
    Result :=
      Result +
      A.Data[I] *
      B.Data[I];
end;

function TensorSum(
  const A: TTensor
): Double;
var
  I: Integer;
begin
  Result := 0.0;

  for I := 0 to High(A.Data) do
    Result :=
      Result + A.Data[I];
end;

function TensorMean(
  const A: TTensor
): Double;
begin
  if TensorSize(A) = 0 then
    raise EArgumentException.Create(
      'Mean of empty tensor'
    );

  Result :=
    TensorSum(A) /
    TensorSize(A);
end;

function TensorMax(
  const A: TTensor
): Double;
var
  I: Integer;
begin
  if TensorSize(A) = 0 then
    raise EArgumentException.Create(
      'Maximum of empty tensor'
    );

  Result := A.Data[0];

  for I := 1 to High(A.Data) do
    if A.Data[I] > Result then
      Result := A.Data[I];
end;

function TensorArgMax(
  const A: TTensor
): Integer;
var
  I: Integer;
begin
  if TensorSize(A) = 0 then
    raise EArgumentException.Create(
      'ArgMax of empty tensor'
    );

  Result := 0;

  for I := 1 to High(A.Data) do
    if A.Data[I] > A.Data[Result] then
      Result := I;
end;

procedure TensorRandomSeed(
  const Seed: Integer
);
begin
  RandSeed := Seed;
end;

function RandomUnit: Double;
begin
  Result := Random;
end;

function RandomGaussian: Double;
var
  U1, U2: Double;
begin
  U1 := RandomUnit;

  if U1 <= 1E-12 then
    U1 := 1E-12;

  U2 := RandomUnit;

  Result :=
    Sqrt(
      -2.0 * Ln(U1)
    ) *
    Cos(
      2.0 * Pi * U2
    );
end;

function TensorRandomUniform(
  const Shape: array of Integer;
  const MinValue, MaxValue: Double
): TTensor;
var
  I: Integer;
begin
  Result :=
    TensorCreate(Shape);

  for I := 0 to High(Result.Data) do
    Result.Data[I] :=
      MinValue +
      RandomUnit *
      (MaxValue - MinValue);
end;

function TensorRandomNormal(
  const Shape: array of Integer;
  const Mean, StdDev: Double
): TTensor;
var
  I: Integer;
begin
  Result :=
    TensorCreate(Shape);

  for I := 0 to High(Result.Data) do
    Result.Data[I] :=
      Mean +
      StdDev *
      RandomGaussian;
end;

function TensorXavierUniform(
  const Rows, Cols: Integer
): TTensor;
var
  Limit: Double;
begin
  Limit :=
    Sqrt(
      6.0 /
      (Rows + Cols)
    );

  Result :=
    TensorRandomUniform(
      [Rows, Cols],
      -Limit,
      Limit
    );
end;

function TensorXavierNormal(
  const Rows, Cols: Integer
): TTensor;
var
  StdDev: Double;
begin
  StdDev :=
    Sqrt(
      2.0 /
      (Rows + Cols)
    );

  Result :=
    TensorRandomNormal(
      [Rows, Cols],
      0.0,
      StdDev
    );
end;

function TensorIsFinite(
  const A: TTensor
): Boolean;
var
  I: Integer;
begin
  Result := True;

  for I := 0 to High(A.Data) do
    if IsNan(A.Data[I]) or
       IsInfinite(A.Data[I]) then
      Exit(False);
end;

function TensorMaxAbs(
  const A: TTensor
): Double;
var
  I: Integer;
  V: Double;
begin
  Result := 0.0;

  for I := 0 to High(A.Data) do
  begin
    V := Abs(A.Data[I]);

    if V > Result then
      Result := V;
  end;
end;

function TensorL2Norm(
  const A: TTensor
): Double;
var
  I: Integer;
  SumSquares: Double;
begin
  SumSquares := 0.0;

  for I := 0 to High(A.Data) do
    SumSquares :=
      SumSquares +
      A.Data[I] *
      A.Data[I];

  Result := Sqrt(SumSquares);
end;

function TensorSameShape(
  const A, B: TTensor
): Boolean;
var
  I: Integer;
begin
  Result :=
    TensorRank(A) =
    TensorRank(B);

  if not Result then
    Exit;

  for I := 0 to TensorRank(A) - 1 do
    if A.Shape[I] <>
       B.Shape[I] then
      Exit(False);
end;

function TensorShapeString(
  const A: TTensor
): string;
var
  I: Integer;
begin
  Result := '[';

  for I := 0 to High(A.Shape) do
  begin
    if I > 0 then
      Result := Result + 'x';

    Result :=
      Result +
      IntToStr(A.Shape[I]);
  end;

  Result := Result + ']';
end;

procedure TensorSave(
  const FileName: string;
  const A: TTensor
);
var
  F: TFileStream;
  Rank: Integer;
  DataCount: Integer;
begin
  F :=
    TFileStream.Create(
      FileName,
      fmCreate
    );

  try
    Rank := TensorRank(A);
    DataCount := TensorSize(A);

    F.WriteBuffer(
      Rank,
      SizeOf(Rank)
    );

    if Rank > 0 then
      F.WriteBuffer(
        A.Shape[0],
        Rank * SizeOf(Integer)
      );

    F.WriteBuffer(
      DataCount,
      SizeOf(DataCount)
    );

    if DataCount > 0 then
      F.WriteBuffer(
        A.Data[0],
        DataCount * SizeOf(Double)
      );
  finally
    F.Free;
  end;
end;

function TensorLoad(
  const FileName: string
): TTensor;
var
  F: TFileStream;
  Rank: Integer;
  DataCount: Integer;
  I: Integer;
begin
  F :=
    TFileStream.Create(
      FileName,
      fmOpenRead or fmShareDenyWrite
    );

  try
    F.ReadBuffer(
      Rank,
      SizeOf(Rank)
    );

    if Rank < 0 then
      raise EInvalidData.Create(
        'Invalid tensor rank'
      );

    SetLength(
      Result.Shape,
      Rank
    );

    if Rank > 0 then
      F.ReadBuffer(
        Result.Shape[0],
        Rank * SizeOf(Integer)
      );

    for I := 0 to Rank - 1 do
      if Result.Shape[I] <= 0 then
        raise EInvalidData.Create(
          'Invalid tensor dimension'
        );

    F.ReadBuffer(
      DataCount,
      SizeOf(DataCount)
    );

    if DataCount <> ProductOfShapeArray(Result.Shape) then
      raise EInvalidData.Create(
        'Tensor data count does not match shape'
      );

    SetLength(
      Result.Data,
      DataCount
    );

    if DataCount > 0 then
      F.ReadBuffer(
        Result.Data[0],
        DataCount * SizeOf(Double)
      );
  finally
    F.Free;
  end;
end;

procedure TensorAssert(
  const Condition: Boolean;
  const Message: string
);
begin
  if not Condition then
    raise EAssertionFailed.Create(
      Message
    );
end;

end.
