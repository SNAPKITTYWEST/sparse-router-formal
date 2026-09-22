-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::formal::route
--
-- Formal model of raw_route. The three-phase structure:
--   Phase 1: region-owner fast path
--   Phase 2: sparse edge traversal (bounded fan-out)
--   Phase 3: least-loaded fallback

import SparseRouter.Types
import SparseRouter.Scoring

namespace SparseRouter.Route

open SparseRouter SparseRouter.Scoring

inductive RouteDecision where
  | ownerFastPath (w : WorkerId)
  | sparseEdgeBest (w : WorkerId) (score : Int)
  | leastLoadedFallback (w : WorkerId)
  deriving Repr

def routeDecisionWorker : RouteDecision → WorkerId
  | .ownerFastPath w => w
  | .sparseEdgeBest w _ => w
  | .leastLoadedFallback w => w

def bestCandidate (workers : Fin MaxWorkers → Worker) (candidates : List WorkerId)
    (t : Task) : Option (WorkerId × Int) :=
  candidates.foldl (fun acc wid =>
    let s := computeScore (workers wid) t
    match acc with
    | none => some (wid, s)
    | some (_, bestScore) => if s > bestScore then some (wid, s) else acc
  ) none

def selectLeastLoaded (workers : Fin MaxWorkers → Worker)
    (n : Nat) (hn : n > 0) (hbound : n ≤ MaxWorkers) : WorkerId :=
  let candidates := List.range n |>.filterMap fun i =>
    if h : i < MaxWorkers then some ⟨i, h⟩ else none
  match candidates with
  | [] => ⟨0, by omega⟩
  | hd :: tl =>
    tl.foldl (fun best wid =>
      if (workers wid).deque.count < (workers best).deque.count
      then wid else best
    ) hd

def rawRoute (state : RouterState) (t : Task)
    (hw : state.numWorkers > 0) : RouteDecision :=
  let r := t.memoryRegion
  let entry := state.regionTable r
  match entry.owner with
  | some owner => .ownerFastPath owner
  | none =>
    let edges := state.sparseEdges r
    match bestCandidate state.workers edges.workers t with
    | some (w, s) => .sparseEdgeBest w s
    | none => .leastLoadedFallback
        (selectLeastLoaded state.workers state.numWorkers hw state.inv_workers_bound)

theorem raw_route_always_returns_worker (state : RouterState) (t : Task)
    (hw : state.numWorkers > 0) :
    ∃ w, routeDecisionWorker (rawRoute state t hw) = w := by
  exact ⟨_, rfl⟩

theorem owner_fast_path_deterministic (state : RouterState) (t : Task)
    (hw : state.numWorkers > 0)
    (howner : (state.regionTable t.memoryRegion).owner = some w) :
    rawRoute state t hw = .ownerFastPath w := by
  simp [rawRoute, howner]

end SparseRouter.Route
