-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::formal::scoring

import SparseRouter.Types

namespace SparseRouter.Scoring

open SparseRouter

def locality (w : Worker) (t : Task) : Int :=
  if w.lastRegion = t.memoryRegion.val then 100 else 0

def memoryAffinity (w : Worker) (t : Task) : Int :=
  if w.cacheFootprint = 0 then 50
  else if w.lastRegion = t.memoryRegion.val then (t.cacheLinesHot : Int) * 10
  else 0

def queuePressure (w : Worker) : Int :=
  if w.deque.count = 0 then 80
  else if w.deque.count < DequeCapacity / 4 then 40
  else if w.deque.count < DequeCapacity / 2 then 0
  else -40

def stealability (w : Worker) : Int :=
  if w.deque.count > 2 then 20 else 0

def migrationCost (w : Worker) (t : Task) : Int :=
  if w.lastRegion = t.memoryRegion.val then 0
  else (t.footprintBytes : Int) / (1024 * 64)

def computeScore (w : Worker) (t : Task) : Int :=
  locality w t + memoryAffinity w t + queuePressure w t +
  stealability w t - migrationCost w t

theorem score_locality_bonus (w : Worker) (t : Task)
    (h : w.lastRegion = t.memoryRegion.val) :
    locality w t = 100 := by
  simp [locality, h]

theorem score_empty_queue_bonus (w : Worker) (t : Task)
    (h : w.deque.count = 0) :
    queuePressure w = 80 := by
  simp [queuePressure, h]

theorem score_no_migration_when_local (w : Worker) (t : Task)
    (h : w.lastRegion = t.memoryRegion.val) :
    migrationCost w t = 0 := by
  simp [migrationCost, h]

end SparseRouter.Scoring
