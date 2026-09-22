-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::formal::types

namespace SparseRouter

def MaxWorkers : Nat := 256
def MaxRegions : Nat := 4096
def MaxSparseEdges : Nat := 16
def DequeCapacity : Nat := 1024

abbrev WorkerId := Fin MaxWorkers
abbrev RegionId := Fin MaxRegions

structure Task where
  memoryRegion : RegionId
  footprintBytes : Nat
  taskId : Nat
  cacheLinesHot : Nat
  priority : Nat
  deriving Repr, BEq

structure WorkDeque where
  tasks : List Task
  count : Nat
  inv_count : count = tasks.length
  inv_cap : count ≤ DequeCapacity

structure Worker where
  id : WorkerId
  deque : WorkDeque
  lastRegion : Nat
  cacheFootprint : Nat
  stealCount : Nat
  localCount : Nat

structure RegionEntry where
  owner : Option WorkerId
  accessCount : Nat
  lastAccessTick : Nat

structure SparseEdgeSet where
  workers : List WorkerId
  count : Nat
  inv_count : count = workers.length
  inv_bounded : count ≤ MaxSparseEdges

structure RouterState where
  workers : Fin MaxWorkers → Worker
  numWorkers : Nat
  inv_workers_bound : numWorkers ≤ MaxWorkers
  regionTable : Fin MaxRegions → RegionEntry
  sparseEdges : Fin MaxRegions → SparseEdgeSet
  globalTick : Nat

end SparseRouter
