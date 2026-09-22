-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::formal::deque

import SparseRouter.Types

namespace SparseRouter.Deque

open SparseRouter

def push (d : WorkDeque) (t : Task) (h : d.count < DequeCapacity) : WorkDeque :=
  { tasks := d.tasks ++ [t]
    count := d.count + 1
    inv_count := by simp [List.length_append]; omega
    inv_cap := by omega }

def pop (d : WorkDeque) (h : d.count > 0) : Task × WorkDeque :=
  let last := d.tasks.getLast (by intro hempty; simp [hempty] at h; exact absurd d.inv_count (by omega))
  let rest := d.tasks.dropLast
  (last,
   { tasks := rest
     count := d.count - 1
     inv_count := by
       simp [List.length_dropLast, d.inv_count]; omega
     inv_cap := by omega })

theorem push_increases_count (d : WorkDeque) (t : Task) (h : d.count < DequeCapacity) :
    (push d t h).count = d.count + 1 := by
  simp [push]

theorem pop_decreases_count (d : WorkDeque) (h : d.count > 0) :
    (pop d h).2.count = d.count - 1 := by
  simp [pop]

theorem push_pop_identity (d : WorkDeque) (t : Task)
    (hcap : d.count < DequeCapacity) :
    let d' := push d t hcap
    let hpos : d'.count > 0 := by simp [push]; omega
    (pop d' hpos).1 = t := by
  simp [push, pop, List.getLast_append]

end SparseRouter.Deque
