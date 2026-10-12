/* SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
 * Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
 * CLONE_GATE: sparse-router-formal::deque
 */

#ifndef SPARSE_ROUTER_DEQUE_H
#define SPARSE_ROUTER_DEQUE_H

#include "types.h"
#include <stddef.h>

static inline bool deque_push(work_deque_t *d, const task_t *t) {
    if (d == NULL || t == NULL || d->count >= DEQUE_CAPACITY ||
        d->tail >= DEQUE_CAPACITY || d->head >= DEQUE_CAPACITY) return false;
    d->tasks[d->tail] = *t;
    d->tail = (d->tail + 1) % DEQUE_CAPACITY;
    d->count++;
    return true;
}

static inline bool deque_pop(work_deque_t *d, task_t *out) {
    if (d == NULL || out == NULL || d->count == 0 || d->count > DEQUE_CAPACITY ||
        d->tail >= DEQUE_CAPACITY || d->head >= DEQUE_CAPACITY) return false;
    d->count--;
    d->tail = (d->tail == 0) ? DEQUE_CAPACITY - 1 : d->tail - 1;
    *out = d->tasks[d->tail];
    return true;
}

static inline bool deque_steal(work_deque_t *d, task_t *out) {
    if (d == NULL || out == NULL || d->count == 0 || d->count > DEQUE_CAPACITY ||
        d->tail >= DEQUE_CAPACITY || d->head >= DEQUE_CAPACITY) return false;
    *out = d->tasks[d->head];
    d->head = (d->head + 1) % DEQUE_CAPACITY;
    d->count--;
    return true;
}

#endif /* SPARSE_ROUTER_DEQUE_H */
