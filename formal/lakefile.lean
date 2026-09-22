-- SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
-- Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
-- CLONE_GATE: sparse-router-formal::lakefile

import Lake
open Lake DSL

package «sparse-router-formal» where
  leanOptions := #[
    ⟨`autoImplicit, false⟩
  ]

@[default_target]
lean_lib «SparseRouter» where
  srcDir := "."
  roots := #[`SparseRouter, `SparseRouter.Types, `SparseRouter.Scoring,
             `SparseRouter.Deque, `SparseRouter.Route, `SparseRouter.Invariants,
             `SparseRouter.Convergence]
