/* SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
 * Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
 * CLONE_GATE: sparse-router-formal::tests
 */

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "../router.h"

#define TEST(name) static void name(void)
#define RUN(name) do { printf("  %-40s", #name); name(); printf("PASS\n"); } while(0)

static router_state_t state;

static void setup(uint32_t nw, uint32_t nr, uint32_t edges) {
    memset(&state, 0, sizeof(state));
    state.num_workers = nw;
    for (uint32_t i = 0; i < nw; i++) state.workers[i].id = i;
    for (uint32_t r = 0; r < nr && r < MAX_REGIONS; r++) {
        uint32_t c = edges < nw ? edges : nw;
        if (c > MAX_SPARSE_EDGES) c = MAX_SPARSE_EDGES;
        state.sparse_edges[r].count = c;
        for (uint32_t i = 0; i < c; i++)
            state.sparse_edges[r].workers[i] = (r + i) % nw;
    }
}

TEST(test_owner_fast_path) {
    setup(4, 8, 2);
    state.region_table[3].has_owner = true;
    state.region_table[3].owner = 2;

    task_t t = { .memory_region = 3, .footprint_bytes = 4096,
                 .task_id = 1, .cache_lines_hot = 4, .priority = 0 };
    worker_id_t w = raw_route(&state, &t);
    assert(w == 2);
    assert(state.workers[2].deque.count == 1);
}

TEST(test_sparse_edge_selection) {
    setup(4, 8, 2);
    task_t t = { .memory_region = 0, .footprint_bytes = 4096,
                 .task_id = 1, .cache_lines_hot = 4, .priority = 0 };
    worker_id_t w = raw_route(&state, &t);
    assert(w != WORKER_NONE);
    assert(state.region_table[0].has_owner);
    assert(state.region_table[0].owner == w);
}

TEST(test_fallback_least_loaded) {
    setup(4, 8, 0);
    task_t t = { .memory_region = 5, .footprint_bytes = 4096,
                 .task_id = 1, .cache_lines_hot = 4, .priority = 0 };
    worker_id_t w = raw_route(&state, &t);
    assert(w < 4);
    assert(state.region_table[5].has_owner);
}

TEST(test_locality_bias) {
    setup(4, 8, 4);
    state.workers[2].last_region = 3;
    task_t t = { .memory_region = 3, .footprint_bytes = 4096,
                 .task_id = 1, .cache_lines_hot = 8, .priority = 0 };
    worker_id_t w = raw_route(&state, &t);
    assert(w == 2);
}

TEST(test_convergence) {
    setup(4, 16, 2);
    for (uint32_t i = 0; i < 100; i++) {
        task_t t = { .memory_region = i % 16, .footprint_bytes = 4096,
                     .task_id = i, .cache_lines_hot = 4, .priority = 0 };
        state.global_tick = i;
        raw_route(&state, &t);
    }
    uint32_t owned = 0;
    for (uint32_t r = 0; r < 16; r++)
        if (state.region_table[r].has_owner) owned++;
    assert(owned == 16);
}

TEST(test_deque_push_pop) {
    work_deque_t d = {0};
    task_t t1 = { .task_id = 42 };
    task_t t2 = { .task_id = 99 };
    assert(deque_push(&d, &t1));
    assert(deque_push(&d, &t2));
    assert(d.count == 2);

    task_t out;
    assert(deque_pop(&d, &out));
    assert(out.task_id == 99);
    assert(deque_pop(&d, &out));
    assert(out.task_id == 42);
    assert(d.count == 0);
}

TEST(test_steal) {
    setup(4, 8, 2);
    for (uint32_t i = 0; i < 10; i++) {
        task_t t = { .memory_region = 0, .task_id = i };
        deque_push(&state.workers[0].deque, &t);
    }
    state.region_table[0].has_owner = true;
    state.region_table[0].owner = 0;
    state.sparse_edges[0].workers[0] = 0;
    state.sparse_edges[0].workers[1] = 1;
    state.sparse_edges[0].count = 2;

    bool stolen = recursive_steal(&state, 1);
    assert(stolen);
    assert(state.workers[1].deque.count == 1);
    assert(state.workers[1].steal_count == 1);
}

int main(void) {
    printf("=== Sparse Router Tests ===\n");
    RUN(test_owner_fast_path);
    RUN(test_sparse_edge_selection);
    RUN(test_fallback_least_loaded);
    RUN(test_locality_bias);
    RUN(test_convergence);
    RUN(test_deque_push_pop);
    RUN(test_steal);
    printf("\nAll tests passed.\n");
    return 0;
}
