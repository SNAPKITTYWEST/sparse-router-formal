/* SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
 * Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
 * CLONE_GATE: sparse-router-formal::types
 */

#ifndef SPARSE_ROUTER_TYPES_H
#define SPARSE_ROUTER_TYPES_H

#include <stdint.h>
#include <stdbool.h>

#define MAX_WORKERS       256
#define MAX_REGIONS       4096
#define MAX_SPARSE_EDGES  16
#define DEQUE_CAPACITY    1024

typedef uint32_t worker_id_t;
typedef uint32_t region_id_t;
typedef int64_t  score_t;

#define WORKER_NONE ((worker_id_t)0xFFFFFFFF)

typedef struct {
    region_id_t  memory_region;
    uint64_t     footprint_bytes;
    uint64_t     task_id;
    uint32_t     cache_lines_hot;
    uint32_t     priority;
} task_t;

typedef struct {
    task_t       tasks[DEQUE_CAPACITY];
    uint32_t     head;
    uint32_t     tail;
    uint32_t     count;
} work_deque_t;

typedef struct {
    worker_id_t  id;
    work_deque_t deque;
    uint64_t     last_region;
    uint64_t     cache_footprint;
    uint32_t     steal_count;
    uint32_t     local_count;
} worker_t;

typedef struct {
    worker_id_t  owner;
    bool         has_owner;
    uint64_t     access_count;
    uint64_t     last_access_tick;
} region_entry_t;

typedef struct {
    worker_id_t  workers[MAX_SPARSE_EDGES];
    uint32_t     count;
} sparse_edge_set_t;

typedef struct {
    worker_t         workers[MAX_WORKERS];
    uint32_t         num_workers;
    region_entry_t   region_table[MAX_REGIONS];
    sparse_edge_set_t sparse_edges[MAX_REGIONS];
    uint64_t         global_tick;
} router_state_t;

#endif /* SPARSE_ROUTER_TYPES_H */
