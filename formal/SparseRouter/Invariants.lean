-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::formal::invariants
--
-- Core safety invariants for the sparse router:
--   INV-1: Every routed task lands on exactly one worker deque
--   INV-2: Region ownership is well-formed (owner < numWorkers)
--   INV-3: Sparse edge sets reference valid workers
--   INV-4: Deque capacity is never exceeded

import SparseRouter.Types
import SparseRouter.Route

namespace SparseRouter.Invariants

open SparseRouter SparseRouter.Route

def RegionOwnerValid (state : RouterState) : Prop :=
  ∀ r : Fin MaxRegions,
    match (state.regionTable r).owner with
    | some w => w.val < state.numWorkers
    | none => True

def SparseEdgesValid (state : RouterState) : Prop :=
  ∀ r : Fin MaxRegions,
    ∀ w ∈ (state.sparseEdges r).workers,
      w.val < state.numWorkers

def DequeCapacityRespected (state : RouterState) : Prop :=
  ∀ i : Fin MaxWorkers,
    (state.workers i).deque.count ≤ DequeCapacity

def TaskCountConserved (state : RouterState) (totalTasks : Nat) : Prop :=
  (Finset.univ.sum fun i : Fin MaxWorkers => (state.workers i).deque.count) = totalTasks

structure WellFormedRouter (state : RouterState) : Prop where
  owner_valid : RegionOwnerValid state
  edges_valid : SparseEdgesValid state
  deque_cap : DequeCapacityRespected state
  workers_pos : state.numWorkers > 0

theorem route_preserves_uniqueness (state : RouterState) (t : Task)
    (hw : state.numWorkers > 0) :
    ∃! w : WorkerId, routeDecisionWorker (rawRoute state t hw) = w := by
  exact ⟨_, rfl, fun _ h => h.symm⟩

theorem sparse_traversal_bounded (edges : SparseEdgeSet) :
    edges.count ≤ MaxSparseEdges :=
  edges.inv_bounded

end SparseRouter.Invariants
