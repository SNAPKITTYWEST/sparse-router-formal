/* SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
 * Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
 * CLONE_GATE: sparse-router-formal::router
 *
 * Sparse Router — locality-aware work-stealing scheduler
 *
 * Novel method: routing decisions traverse only the sparse edge set
 * (bounded fan-out per memory region) rather than scanning all workers.
 * Successful routing enqueues exactly once. WORKER_NONE leaves the task
 * with the caller. The Lean decision model is separate from this executable;
 * no implementation-refinement or scheduler-liveness proof is claimed.
 */

#ifndef SPARSE_ROUTER_H
#define SPARSE_ROUTER_H

#include "types.h"
#include "scoring.h"
#include "deque.h"

static inline worker_id_t select_least_loaded(const router_state_t *state) {
    worker_id_t best = WORKER_NONE;
    uint32_t min_load = UINT32_MAX;
    for (uint32_t i = 0; i < state->num_workers; i++) {
        if (state->workers[i].deque.count < DEQUE_CAPACITY &&
            state->workers[i].deque.count < min_load) {
            min_load = state->workers[i].deque.count;
            best = i;
        }
    }
    return best;
}

static inline worker_id_t raw_route(router_state_t *state, task_t *task) {
    /* WORKER_NONE means no task was accepted: caller retains it for retry.
       Callers must externally serialize access to this reference scheduler. */
    if (state == NULL || task == NULL || state->num_workers == 0 ||
        state->num_workers > MAX_WORKERS || task->memory_region >= MAX_REGIONS)
        return WORKER_NONE;
    region_id_t r = task->memory_region;
    const sparse_edge_set_t *candidates = &state->sparse_edges[r];
    if (candidates->count > MAX_SPARSE_EDGES) return WORKER_NONE;
    for (uint32_t i = 0; i < candidates->count; i++)
        if (candidates->workers[i] >= state->num_workers) return WORKER_NONE;
    if (state->region_table[r].has_owner &&
        state->region_table[r].owner >= state->num_workers) return WORKER_NONE;

    /* Phase 1: region-owner fast path */
    if (state->region_table[r].has_owner) {
        worker_id_t owner = state->region_table[r].owner;
        if (deque_push(&state->workers[owner].deque, task)) {
            state->workers[owner].local_count++;
            state->region_table[r].access_count++;
            state->region_table[r].last_access_tick = state->global_tick;
            return owner;
        }
    }

    /* Phase 2: sparse edge traversal — O(MAX_SPARSE_EDGES) not O(num_workers) */
    worker_id_t best = WORKER_NONE;
    score_t best_score = INT64_MIN;

    for (uint32_t i = 0; i < candidates->count; i++) {
        worker_id_t wid = candidates->workers[i];
        if (state->workers[wid].deque.count >= DEQUE_CAPACITY) continue;
        score_t s = compute_score(&state->workers[wid], task);
        if (s > best_score) {
            best_score = s;
            best = wid;
        }
    }

    if (best != WORKER_NONE && deque_push(&state->workers[best].deque, task)) {
        state->workers[best].local_count++;
        state->region_table[r].owner = best;
        state->region_table[r].has_owner = true;
        state->region_table[r].access_count = 1;
        state->region_table[r].last_access_tick = state->global_tick;
        return best;
    }

    /* Phase 3: global fallback — no sparse edges for this region */
    worker_id_t fallback = select_least_loaded(state);
    if (fallback == WORKER_NONE || !deque_push(&state->workers[fallback].deque, task))
        return WORKER_NONE;
    state->workers[fallback].local_count++;
    state->region_table[r].owner = fallback;
    state->region_table[r].has_owner = true;
    state->region_table[r].access_count = 1;
    state->region_table[r].last_access_tick = state->global_tick;
    return fallback;
}

/* Recursive steal chain: if a worker is idle, steal from the
   highest-loaded sparse neighbor, recursively rebalancing. */
static inline bool recursive_steal(router_state_t *state, worker_id_t thief) {
    if (state == NULL || state->num_workers == 0 || state->num_workers > MAX_WORKERS ||
        thief >= state->num_workers) return false;
    worker_t *tw = &state->workers[thief];
    if (tw->deque.head >= DEQUE_CAPACITY || tw->deque.tail >= DEQUE_CAPACITY) return false;
    if (tw->deque.count > 0) return false;

    worker_id_t victim = WORKER_NONE;
    uint32_t max_load = 0;

    for (uint32_t r = 0; r < MAX_REGIONS; r++) {
        if (!state->region_table[r].has_owner) continue;
        if (state->region_table[r].owner >= state->num_workers) continue;
        if (state->region_table[r].owner == thief) continue;

        const sparse_edge_set_t *edges = &state->sparse_edges[r];
        if (edges->count > MAX_SPARSE_EDGES) continue;
        for (uint32_t i = 0; i < edges->count; i++) {
            if (edges->workers[i] == thief) {
                worker_id_t owner = state->region_table[r].owner;
                if (state->workers[owner].deque.count > max_load) {
                    max_load = state->workers[owner].deque.count;
                    victim = owner;
                }
                break;
            }
        }
    }

    if (victim == WORKER_NONE) {
        for (uint32_t i = 0; i < state->num_workers; i++) {
            if (i == thief) continue;
            if (state->workers[i].deque.count > max_load) {
                max_load = state->workers[i].deque.count;
                victim = i;
            }
        }
    }

    if (victim == WORKER_NONE || max_load < 2) return false;

    task_t stolen;
    if (!deque_steal(&state->workers[victim].deque, &stolen)) return false;

    deque_push(&tw->deque, &stolen);
    tw->steal_count++;
    return true;
}

#endif /* SPARSE_ROUTER_H */
