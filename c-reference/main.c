/* SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
 * Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
 * CLONE_GATE: sparse-router-formal::main
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "router.h"

static void init_state(router_state_t *s, uint32_t num_workers) {
    memset(s, 0, sizeof(*s));
    s->num_workers = num_workers;
    for (uint32_t i = 0; i < num_workers; i++) {
        s->workers[i].id = i;
    }
}

static void seed_sparse_edges(router_state_t *s, uint32_t num_regions,
                               uint32_t edges_per_region) {
    for (uint32_t r = 0; r < num_regions; r++) {
        uint32_t count = edges_per_region;
        if (count > s->num_workers) count = s->num_workers;
        if (count > MAX_SPARSE_EDGES) count = MAX_SPARSE_EDGES;
        s->sparse_edges[r].count = count;
        for (uint32_t i = 0; i < count; i++) {
            s->sparse_edges[r].workers[i] = (r + i) % s->num_workers;
        }
    }
}

static void bench_routing(uint32_t num_workers, uint32_t num_tasks,
                           uint32_t num_regions, uint32_t edges) {
    router_state_t *state = calloc(1, sizeof(router_state_t));
    if (!state) { perror("calloc"); return; }

    init_state(state, num_workers);
    seed_sparse_edges(state, num_regions, edges);

    struct timespec t0, t1;
    timespec_get(&t0, TIME_UTC);

    for (uint32_t i = 0; i < num_tasks; i++) {
        task_t t = {
            .memory_region  = i % num_regions,
            .footprint_bytes = 4096 + (i % 16) * 4096,
            .task_id        = i,
            .cache_lines_hot = 4 + (i % 12),
            .priority       = i % 4,
        };
        state->global_tick = i;
        raw_route(state, &t);
    }

    timespec_get(&t1, TIME_UTC);
    double elapsed = (t1.tv_sec - t0.tv_sec) +
                     (t1.tv_nsec - t0.tv_nsec) / 1e9;

    printf("workers=%u tasks=%u regions=%u edges=%u\n",
           num_workers, num_tasks, num_regions, edges);
    printf("  elapsed: %.6f s  (%.0f routes/sec)\n",
           elapsed, num_tasks / elapsed);

    uint32_t total = 0, max_q = 0, min_q = UINT32_MAX;
    for (uint32_t i = 0; i < num_workers; i++) {
        uint32_t c = state->workers[i].deque.count;
        total += c;
        if (c > max_q) max_q = c;
        if (c < min_q) min_q = c;
    }
    printf("  total queued: %u  min/max: %u/%u  balance: %.2f%%\n",
           total, min_q, max_q,
           num_workers > 0 ? 100.0 * min_q / (max_q ? max_q : 1) : 0.0);

    uint32_t owned = 0;
    for (uint32_t r = 0; r < num_regions; r++) {
        if (state->region_table[r].has_owner) owned++;
    }
    printf("  regions owned: %u/%u\n\n", owned, num_regions);

    free(state);
}

int main(void) {
    printf("=== Sparse Router Benchmark ===\n\n");
    bench_routing(8, 100000, 64, 4);
    bench_routing(16, 100000, 256, 4);
    bench_routing(64, 1000000, 1024, 8);
    bench_routing(256, 1000000, 4096, 16);
    return 0;
}
