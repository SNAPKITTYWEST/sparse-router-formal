/* SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
 * Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
 * CLONE_GATE: sparse-router-formal::scoring
 */

#ifndef SPARSE_ROUTER_SCORING_H
#define SPARSE_ROUTER_SCORING_H

#include "types.h"

static inline score_t locality(const worker_t *w, const task_t *t) {
    return (w->last_region == t->memory_region) ? 100 : 0;
}

static inline score_t memory_affinity(const worker_t *w, const task_t *t) {
    if (w->cache_footprint == 0) return 50;
    uint64_t overlap = (w->last_region == t->memory_region)
        ? t->cache_lines_hot : 0;
    return (score_t)(overlap * 10);
}

static inline score_t queue_pressure(const worker_t *w) {
    if (w->deque.count == 0) return 80;
    if (w->deque.count < DEQUE_CAPACITY / 4) return 40;
    if (w->deque.count < DEQUE_CAPACITY / 2) return 0;
    return -40;
}

static inline score_t stealability(const worker_t *w) {
    return (w->deque.count > 2) ? 20 : 0;
}

static inline score_t migration_cost(const worker_t *w, const task_t *t) {
    if (w->last_region == t->memory_region) return 0;
    return (score_t)(t->footprint_bytes / (1024 * 64));
}

static inline score_t compute_score(const worker_t *w, const task_t *t) {
    return locality(w, t)
         + memory_affinity(w, t)
         + queue_pressure(w)
         + stealability(w)
         - migration_cost(w, t);
}

#endif /* SPARSE_ROUTER_SCORING_H */
