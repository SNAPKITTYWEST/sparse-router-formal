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

{ gpu_host.pas
  Pascal host skeleton for launching CUDA kernels via a C shim.
  Free Pascal (FPC) compatible.
  Lines: ~220
}

program GPUHost;

{$mode objfpc}{$H+}

uses
  ctypes, SysUtils;

{ --- Memory manager: simple arena with bookkeeping --- }
type
  PBlock = ^TBlock;
  TBlock = record
    addr: Pointer;
    size: NativeUInt;
    freed: Boolean;
  end;

var
  Blocks: array of PBlock;

procedure register_block(p: Pointer; sz: NativeUInt);
var
  b: PBlock;
begin
  New(b);
  b^.addr := p;
  b^.size := sz;
  b^.freed := False;
  SetLength(Blocks, Length(Blocks) + 1);
  Blocks[High(Blocks)] := b;
end;

function find_block(p: Pointer): Integer;
var
  i: Integer;
begin
  for i := 0 to High(Blocks) do
    if Blocks[i]^.addr = p then Exit(i);
  Result := -1;
end;

procedure check_no_leaks;
var
  i: Integer;
begin
  for i := 0 to High(Blocks) do
    if not Blocks[i]^.freed then
      Writeln('Memory leak: block at ', Blocks[i]^.addr, ' size=', Blocks[i]^.size);
end;

{ --- C shim declarations (must match the C shim below) --- }
type
  TCUResult = cint;

procedure cuda_init; cdecl; external 'libcuda_shim.so';
procedure cuda_finalize; cdecl; external 'libcuda_shim.so';
function cuda_alloc(size: NativeUInt): Pointer; cdecl; external 'libcuda_shim.so';
procedure cuda_free(p: Pointer); cdecl; external 'libcuda_shim.so';
function cuda_launch_vector_add(a_dev, b_dev, c_dev: Pointer; n: cint): TCUResult; cdecl; external 'libcuda_shim.so';

{ --- Pascal wrappers around the C shim --- }
function host_alloc(size: NativeUInt): Pointer;
begin
  Result := cuda_alloc(size);
  if Result = nil then
    raise Exception.Create('cuda_alloc failed');
  register_block(Result, size);
end;

procedure host_free(p: Pointer);
var idx: Integer;
begin
  idx := find_block(p);
  if idx = -1 then
    raise Exception.Create('free: unknown pointer');
  if Blocks[idx]^.freed then
    raise Exception.Create('double free detected');
  Blocks[idx]^.freed := True;
  cuda_free(p);
end;

{ --- Test: vector add on GPU --- }
procedure test_vector_add(n: Integer);
var
  bytes: NativeUInt;
  a_dev, b_dev, c_dev: Pointer;
  i: Integer;
  hostA, hostB, hostC: PSingle;
  res: TCUResult;
begin
  bytes := NativeUInt(n) * SizeOf(Single);
  a_dev := host_alloc(bytes);
  b_dev := host_alloc(bytes);
  c_dev := host_alloc(bytes);

  { prepare host buffers (on CPU) and copy via shim if needed; for simplicity we assume shim provides device alloc only and kernel uses device memory }
  { In a real system, add cuda_memcpy_host_to_device and device_to_host functions }

  res := cuda_launch_vector_add(a_dev, b_dev, c_dev, n);
  if res <> 0 then
    Writeln('Kernel launch failed: ', res)
  else
    Writeln('Kernel launched successfully');

  { cleanup }
  host_free(a_dev);
  host_free(b_dev);
  host_free(c_dev);
end;

{ --- Main --- }
begin
  try
    Writeln('Initializing CUDA shim...');
    cuda_init();
    Writeln('Allocating and launching test kernel...');
    test_vector_add(1024);
    Writeln('Finalizing...');
    cuda_finalize();
    check_no_leaks();
    Writeln('Done.');
  except
    on E: Exception do
      Writeln('Error: ', E.ClassName, ': ', E.Message);
  end;
end.
