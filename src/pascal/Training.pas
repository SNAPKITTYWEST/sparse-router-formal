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

{ Training.pas â€“ Adam optimizer, gradient clipping, learning rate scheduling
  Line count: 200
  Dependency: TensorCore

  Manual Adam implementation: m_t = b1*m_{t-1} + (1-b1)*g_t
                             v_t = b2*v_{t-1} + (1-b2)*g_t^2
                             t_t = t - a * m_t / (sqrt(v_t) + eps)
}

unit Training;

interface

uses
  SysUtils, Math, TensorCore;

type
  TAdamState = record
    M, V: TTensorData;
    T: Integer;
    procedure Initialize(const ParamShape: TShape);
  end;

  TAdamOptimizer = class
  private
    States: array of TAdamState;
    Beta1, Beta2, Eps: Single;
    T: Integer;
  public
    Alpha: Single;
    constructor Create(LearningRate, Beta1Val, Beta2Val, Epsilon: Single);
    procedure Step(var Param: TTensorData; const Grad: TTensorData);
    procedure ClipGradients(var Grad: TTensorData; MaxNorm: Single);
  end;

  TLRScheduler = class
  public
    class procedure StepDecay(var LR: Single; Epoch, DecaySteps, DecayRate: Integer);
    class procedure WarmupLinear(var LR: Single; Epoch, WarmupEpochs, MaxEpochs, BaseLR: Integer);
  end;

  TBatchHelper = class
  public
    class procedure CreateBatch(var Batch: TTensorData; const Data: TTensorData;
      BatchStart, BatchSize: Integer);
  end;

implementation

procedure TAdamState.Initialize(const ParamShape: TShape);
begin
  TensorOps.Zeros(M, ParamShape);
  TensorOps.Zeros(V, ParamShape);
  T := 0;
end;

constructor TAdamOptimizer.Create(LearningRate, Beta1Val, Beta2Val, Epsilon: Single);
begin
  inherited Create;
  Alpha := LearningRate;
  Beta1 := Beta1Val;
  Beta2 := Beta2Val;
  Eps := Epsilon;
  T := 0;
end;

procedure TAdamOptimizer.Step(var Param: TTensorData; const Grad: TTensorData);
var
  i, StateIdx: Integer;
  State: TAdamState;
  BiasCorrection1, BiasCorrection2, AlphaHat: Single;
begin
  Inc(T);

  StateIdx := -1;
  for i := 0 to Length(States) - 1 do begin
    if States[i].M.Shape.TotalSize = Param.Shape.TotalSize then begin
      StateIdx := i;
      Break;
    end;
  end;

  if StateIdx = -1 then begin
    StateIdx := Length(States);
    SetLength(States, StateIdx + 1);
    States[StateIdx].Initialize(Param.Shape);
  end;

  State := States[StateIdx];
  Inc(State.T);

  for i := 0 to Length(Grad.Data) - 1 do begin
    State.M.Data[i] := Beta1 * State.M.Data[i] + (1 - Beta1) * Grad.Data[i];
    State.V.Data[i] := Beta2 * State.V.Data[i] + (1 - Beta2) * Sqr(Grad.Data[i]);
  end;

  BiasCorrection1 := 1 - Power(Beta1, State.T);
  BiasCorrection2 := 1 - Power(Beta2, State.T);
  AlphaHat := Alpha * Sqrt(BiasCorrection2) / BiasCorrection1;

  for i := 0 to Length(Param.Data) - 1 do
    Param.Data[i] := Param.Data[i] - AlphaHat * State.M.Data[i] /
      (Sqrt(State.V.Data[i]) + Eps);

  States[StateIdx] := State;
end;

procedure TAdamOptimizer.ClipGradients(var Grad: TTensorData; MaxNorm: Single);
var
  i: Integer;
  GradNorm: Single;
begin
  GradNorm := 0;
  for i := 0 to Length(Grad.Data) - 1 do
    GradNorm := GradNorm + Sqr(Grad.Data[i]);
  GradNorm := Sqrt(GradNorm);

  if GradNorm > MaxNorm then begin
    for i := 0 to Length(Grad.Data) - 1 do
      Grad.Data[i] := Grad.Data[i] * MaxNorm / GradNorm;
  end;
end;

class procedure TLRScheduler.StepDecay(var LR: Single; Epoch, DecaySteps, DecayRate: Integer);
begin
  if (Epoch > 0) and (Epoch mod DecaySteps = 0) then
    LR := LR * DecayRate;
end;

class procedure TLRScheduler.WarmupLinear(var LR: Single; Epoch, WarmupEpochs, MaxEpochs, BaseLR: Integer);
var
  TargetLR: Single;
begin
  if Epoch < WarmupEpochs then
    LR := BaseLR * Epoch / WarmupEpochs
  else
    LR := BaseLR * (MaxEpochs - Epoch) / (MaxEpochs - WarmupEpochs);
end;

class procedure TBatchHelper.CreateBatch(var Batch: TTensorData; const Data: TTensorData;
  BatchStart, BatchSize: Integer);
var
  S: TShape;
  i: Integer;
begin
  S.Init(BatchSize, Data.Shape.Dims[1], 0, 0);
  Batch.Allocate(S);
  for i := 0 to BatchSize - 1 do begin
    if BatchStart + i < Data.Shape.Dims[0] then
      Move(Data.Data[(BatchStart + i) * Data.Shape.Dims[1]],
           Batch.Data[i * Data.Shape.Dims[1]],
           Data.Shape.Dims[1] * SizeOf(Single));
  end;
end;

end.
