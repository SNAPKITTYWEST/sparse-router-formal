{ Ownership State Machine: Manages OWNED, SHARED, EXCLUSIVE states.
  Tracks ownership tokens and implements governance state transitions.

  State Machine:
    - OWNED: Exclusive ownership, single reference, can mutate
    - SHARED: Multiple references, immutable (COW semantics)
    - EXCLUSIVE: Recovered exclusive after materialization

  Transitions:
    OWNED --[clone]--> SHARED
    SHARED --[materialize]--> EXCLUSIVE
    EXCLUSIVE --[release-share]--> OWNED
}

unit ownership;

{$MODE FPC}

interface

type
  { Ownership state enumeration }
  TOwnershipState = (
    OS_OWNED,      { Exclusive ownership }
    OS_SHARED,     { Multiple references }
    OS_EXCLUSIVE   { Recovered exclusive after materialization }
  );

  { Ownership token }
  TOwnershipToken = record
    state: TOwnershipState;
    generation: uint32;
    refcount: uint32;
  end;

{ Create initial ownership token (OWNED state) }
function CreateOwnershipToken(generation: uint32): TOwnershipToken;

{ Transition to SHARED state (on clone) }
function TransitionToShared(token: TOwnershipToken): TOwnershipToken;

{ Transition to EXCLUSIVE state (on materialize) }
function TransitionToExclusive(token: TOwnershipToken; new_generation: uint32): TOwnershipToken;

{ Release shared reference, potentially revert to OWNED }
function ReleaseSharedReference(
  token: TOwnershipToken;
  refcount: uint32
): TOwnershipToken;

{ Check if state allows mutation }
function CanMutate(state: TOwnershipState): boolean;

{ Check if state allows viewing }
function CanView(state: TOwnershipState): boolean;

{ Get string representation for debugging }
function StateToString(state: TOwnershipState): string;

{ Check state consistency }
function IsValidTransition(from_state, to_state: TOwnershipState): boolean;

implementation

function CreateOwnershipToken(generation: uint32): TOwnershipToken;
begin
  with result do
  begin
    state := OS_OWNED;
    generation := generation;
    refcount := 1;
  end;
end;

function TransitionToShared(token: TOwnershipToken): TOwnershipToken;
begin
  result := token;
  if token.state = OS_OWNED then
  begin
    result.state := OS_SHARED;
    inc(result.refcount);
  end;
end;

function TransitionToExclusive(token: TOwnershipToken; new_generation: uint32): TOwnershipToken;
begin
  result := token;
  if token.state = OS_SHARED then
  begin
    result.state := OS_EXCLUSIVE;
    result.generation := new_generation;
  end;
end;

function ReleaseSharedReference(
  token: TOwnershipToken;
  refcount: uint32
): TOwnershipToken;
begin
  result := token;
  if token.state = OS_SHARED then
  begin
    dec(result.refcount);
    if refcount = 1 then
      result.state := OS_OWNED;
  end;
end;

function CanMutate(state: TOwnershipState): boolean;
begin
  result := (state = OS_OWNED) or (state = OS_EXCLUSIVE);
end;

function CanView(state: TOwnershipState): boolean;
begin
  result := true; { Any state allows viewing }
end;

function StateToString(state: TOwnershipState): string;
begin
  case state of
    OS_OWNED:
      result := 'OWNED';
    OS_SHARED:
      result := 'SHARED';
    OS_EXCLUSIVE:
      result := 'EXCLUSIVE';
  else
    result := 'UNKNOWN';
  end;
end;

function IsValidTransition(from_state, to_state: TOwnershipState): boolean;
begin
  case from_state of
    OS_OWNED:
      result := (to_state = OS_SHARED) or (to_state = OS_OWNED);
    OS_SHARED:
      result := (to_state = OS_EXCLUSIVE) or (to_state = OS_SHARED);
    OS_EXCLUSIVE:
      result := (to_state = OS_OWNED) or (to_state = OS_EXCLUSIVE);
  else
    result := false;
  end;
end;

end.
