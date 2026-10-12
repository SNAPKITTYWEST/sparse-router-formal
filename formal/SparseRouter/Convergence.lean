-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::formal::convergence
-- The decision model does not update state or establish scheduler liveness.
import SparseRouter.Types
import SparseRouter.Route
import SparseRouter.Invariants

namespace SparseRouter.Convergence
open SparseRouter SparseRouter.Route

def unownedRegions (state : RouterState) : Nat :=
  (List.range MaxRegions).countP fun i =>
    if h : i < MaxRegions then (state.regionTable ⟨i, h⟩).owner = none else false

def potential (state : RouterState) : Nat := unownedRegions state

theorem owner_fast_path_preserves_potential (state : RouterState) (t : Task)
    (hw : state.numWorkers > 0)
    (howner : (state.regionTable t.memoryRegion).owner ≠ none) :
    ∃ w, rawRoute state t hw = .ownerFastPath w := by
  simp [rawRoute]
  match h : (state.regionTable t.memoryRegion).owner with
  | some w => exact ⟨w, rfl⟩
  | none => exact absurd h howner

theorem sparse_edge_assigns_owner (state : RouterState) (t : Task)
    (hw : state.numWorkers > 0)
    (hunowned : (state.regionTable t.memoryRegion).owner = none)
    (hcand : bestCandidate state.workers (state.sparseEdges t.memoryRegion).workers t = some (w, s)) :
    rawRoute state t hw = .sparseEdgeBest w s := by
  simp [rawRoute, hunowned, hcand]

theorem potential_bounded : ∀ state : RouterState,
    potential state ≤ MaxRegions := by
  intro state
  unfold potential unownedRegions
  exact Nat.le_trans List.countP_le_length (by simp)

theorem zero_potential_all_regions_owned :
    ∀ state : RouterState, potential state = 0 →
    ∀ r : Fin MaxRegions, (state.regionTable r).owner ≠ none := by
  intro state hpot r
  simp [potential, unownedRegions] at hpot
  exact hpot r.val r.isLt r.isLt

end SparseRouter.Convergence
