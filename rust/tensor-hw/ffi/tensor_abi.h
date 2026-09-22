/*
 * NraayTensor FFI ABI Contract
 *
 * C-compatible, hardware-oriented tensor descriptor.
 * Shared between Rust (tensor.rs) and Pascal (tensor.pas).
 *
 * This is the canonical binary protocol for all Rust↔Pascal communication.
 */

#ifndef TENSOR_ABI_H
#define TENSOR_ABI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Tensor Descriptor — C-compatible packed structure
 * ============================================================================
 */

typedef struct {
    /* Buffer identity and address */
    uint64_t buffer_id;
    uint64_t base_addr;
    uint64_t offset;

    /* Rank and element count */
    uint32_t rank;
    uint64_t element_count;

    /* Shape and stride pointers (owned by tensor runtime, valid only during call) */
    uint64_t shape_ptr;
    uint64_t stride_ptr;

    /* Governance and state */
    uint8_t governance;        /* B=0 (shared), B=1 (exclusive) */
    uint8_t ownership;         /* OWNED=0, SHARED=1, EXCLUSIVE=2 */
    uint8_t materialized;      /* 0=not materialized, 1=materialized */
    uint8_t state;             /* internal state byte */

    /* Generation counter for stale-descriptor detection */
    uint64_t generation;
} tensor_descriptor_t;

/* ============================================================================
 * Ownership State
 * ============================================================================
 */

typedef enum {
    OWNERSHIP_OWNED = 0,
    OWNERSHIP_SHARED = 1,
    OWNERSHIP_EXCLUSIVE = 2,
} ownership_state_t;

/* ============================================================================
 * Governance State
 * ============================================================================
 */

typedef enum {
    GOVERNANCE_SHARED = 0,      /* B=0: shared/non-canonical */
    GOVERNANCE_CANONICAL = 1,   /* B=1: canonical/contiguous */
} governance_state_t;

/* ============================================================================
 * Operation Result
 * ============================================================================
 */

typedef struct {
    int32_t success;
    uint32_t error_code;
    char error_msg[256];
} tensor_result_t;

typedef struct {
    tensor_descriptor_t descriptor;
    tensor_result_t result;
} tensor_op_result_t;

/* ============================================================================
 * FFI Operations
 * ============================================================================
 */

/* Allocate a new tensor buffer with given shape */
tensor_op_result_t tensor_allocate(
    const uint32_t rank,
    const uint64_t *shape,
    uint64_t element_size
);

/* Deallocate a tensor */
tensor_result_t tensor_deallocate(const tensor_descriptor_t *desc);

/* Create a view (slice) of a tensor */
tensor_op_result_t tensor_view(
    const tensor_descriptor_t *source,
    const uint64_t *start_indices,
    const uint64_t *end_indices
);

/* Create a transposed view (no data copy) */
tensor_op_result_t tensor_transpose(
    const tensor_descriptor_t *source,
    const uint32_t *permutation
);

/* Create a reshaped view (if possible without data copy) */
tensor_op_result_t tensor_reshape(
    const tensor_descriptor_t *source,
    const uint32_t new_rank,
    const uint64_t *new_shape
);

/* Materialize a non-contiguous view into a new contiguous buffer */
tensor_op_result_t tensor_materialize(
    const tensor_descriptor_t *source
);

/* Clone a tensor (deep copy or independent view depending on state) */
tensor_op_result_t tensor_clone(
    const tensor_descriptor_t *source
);

/* Load element value */
typedef struct {
    float value;
    tensor_result_t result;
} tensor_load_result_t;

tensor_load_result_t tensor_load(
    const tensor_descriptor_t *desc,
    const uint64_t *indices
);

/* Store element value */
tensor_result_t tensor_store(
    const tensor_descriptor_t *desc,
    const uint64_t *indices,
    float value
);

/* Validate descriptor state (generation, ownership, bounds) */
tensor_result_t tensor_validate(
    const tensor_descriptor_t *desc
);

/* Get descriptor info as human-readable string (for debugging) */
typedef struct {
    char buffer[512];
    tensor_result_t result;
} tensor_info_t;

tensor_info_t tensor_info(
    const tensor_descriptor_t *desc
);

/* ============================================================================
 * Memory and Ownership Inspection (for conformance testing)
 * ============================================================================
 */

/* Get current generation counter for a buffer ID */
typedef struct {
    uint64_t generation;
    tensor_result_t result;
} tensor_gen_result_t;

tensor_gen_result_t tensor_buffer_generation(uint64_t buffer_id);

/* Get strong reference count for a buffer */
typedef struct {
    uint64_t count;
    tensor_result_t result;
} tensor_refcount_t;

tensor_refcount_t tensor_buffer_refcount(uint64_t buffer_id);

/* ============================================================================
 * Governance State Inspection
 * ============================================================================
 */

/* Get governance state */
typedef struct {
    uint8_t state;
    tensor_result_t result;
} tensor_gov_result_t;

tensor_gov_result_t tensor_governance_state(const tensor_descriptor_t *desc);

/* ============================================================================
 * Conformance Oracle (Rust validates what Pascal computed)
 * ============================================================================
 */

/* Verify that two descriptors represent the same logical tensor */
typedef struct {
    int32_t match;
    char discrepancy[256];
    tensor_result_t result;
} tensor_compare_t;

tensor_compare_t tensor_compare_descriptors(
    const tensor_descriptor_t *rust,
    const tensor_descriptor_t *pascal
);

/* Verify address calculation matches */
typedef struct {
    int32_t match;
    uint64_t rust_addr;
    uint64_t pascal_addr;
    tensor_result_t result;
} tensor_addr_compare_t;

tensor_addr_compare_t tensor_compare_address(
    const tensor_descriptor_t *desc,
    const uint64_t *indices
);

#ifdef __cplusplus
}
#endif

#endif /* TENSOR_ABI_H */
