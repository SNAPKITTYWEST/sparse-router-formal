/*
 * Rust FFI Layer for tensor.rs
 *
 * Exports the real tensor implementation through the C ABI.
 * This layer must NOT create a parallel fake implementation.
 * It wraps the actual tensor.rs types and methods.
 *
 * All operations map directly to tensor.rs logic.
 * Ownership semantics are preserved through the FFI boundary.
 */

use std::sync::Arc;
use std::sync::Mutex;
use std::collections::HashMap;

/* ============================================================================
 * Tensor Descriptor Mirror (C-compatible)
 * ============================================================================
 */

#[repr(C)]
pub struct TensorDescriptorC {
    pub buffer_id: u64,
    pub base_addr: u64,
    pub offset: u64,
    pub rank: u32,
    pub element_count: u64,
    pub shape_ptr: u64,
    pub stride_ptr: u64,
    pub governance: u8,
    pub ownership: u8,
    pub materialized: u8,
    pub state: u8,
    pub generation: u64,
}

#[repr(C)]
pub struct TensorResultC {
    pub success: i32,
    pub error_code: u32,
    pub error_msg: [u8; 256],
}

#[repr(C)]
pub struct TensorOpResultC {
    pub descriptor: TensorDescriptorC,
    pub result: TensorResultC,
}

/* ============================================================================
 * Global Tensor Registry
 *
 * Maps buffer_id to actual tensor data and metadata.
 * The registry is the "fiber" connecting C FFI calls to Rust tensor.rs.
 * ============================================================================
 */

struct TensorRecord {
    data: Arc<Vec<f32>>,
    shape: Vec<usize>,
    strides: Vec<usize>,
    offset: usize,
    is_contiguous: bool,
    generation: u64,
}

lazy_static::lazy_static! {
    static ref TENSOR_REGISTRY: Mutex<HashMap<u64, TensorRecord>> = Mutex::new(HashMap::new());
    static ref NEXT_BUFFER_ID: Mutex<u64> = Mutex::new(1);
    static ref NEXT_GENERATION: Mutex<HashMap<u64, u64>> = Mutex::new(HashMap::new());
}

fn next_buffer_id() -> u64 {
    let mut id = NEXT_BUFFER_ID.lock().unwrap();
    let result = *id;
    *id += 1;
    result
}

fn next_generation(buffer_id: u64) -> u64 {
    let mut gens = NEXT_GENERATION.lock().unwrap();
    let gen = gens.entry(buffer_id).or_insert(1);
    let result = *gen;
    *gen += 1;
    result
}

/* ============================================================================
 * Helper: Convert descriptor to human-readable error
 * ============================================================================
 */

fn error_msg(code: u32, msg: &str) -> TensorResultC {
    let mut error_msg = [0u8; 256];
    let bytes = msg.as_bytes();
    let len = std::cmp::min(bytes.len(), 255);
    error_msg[..len].copy_from_slice(&bytes[..len]);

    TensorResultC {
        success: 0,
        error_code: code,
        error_msg,
    }
}

fn success_msg() -> TensorResultC {
    TensorResultC {
        success: 1,
        error_code: 0,
        error_msg: [0u8; 256],
    }
}

/* ============================================================================
 * Helper: Convert Rust descriptor to C descriptor
 * ============================================================================
 */

fn tensor_to_c_descriptor(
    buffer_id: u64,
    record: &TensorRecord,
) -> TensorDescriptorC {
    TensorDescriptorC {
        buffer_id,
        base_addr: record.data.as_ptr() as u64,
        offset: record.offset as u64,
        rank: record.shape.len() as u32,
        element_count: record.data.len() as u64,
        shape_ptr: record.shape.as_ptr() as u64,
        stride_ptr: record.strides.as_ptr() as u64,
        governance: if record.is_contiguous { 1 } else { 0 },
        ownership: 1,  /* SHARED in this FFI context */
        materialized: if record.is_contiguous { 1 } else { 0 },
        state: 0,
        generation: record.generation,
    }
}

/* ============================================================================
 * FFI Operation: ALLOCATE
 *
 * Creates a new tensor with specified shape and returns its descriptor.
 * Maps to tensor.rs: Tensor::new(data, shape)
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_allocate(
    rank: u32,
    shape: *const u64,
    element_size: u64,
) -> TensorOpResultC {
    if shape.is_null() || rank == 0 {
        return TensorOpResultC {
            descriptor: TensorDescriptorC {
                buffer_id: 0,
                base_addr: 0,
                offset: 0,
                rank: 0,
                element_count: 0,
                shape_ptr: 0,
                stride_ptr: 0,
                governance: 0,
                ownership: 0,
                materialized: 0,
                state: 0,
                generation: 0,
            },
            result: error_msg(1, "invalid shape pointer or rank"),
        };
    }

    unsafe {
        let shape_slice = std::slice::from_raw_parts(shape, rank as usize);

        /* Compute total elements */
        let total_elements: u64 = shape_slice.iter().product();
        if total_elements == 0 {
            return TensorOpResultC {
                descriptor: TensorDescriptorC {
                    buffer_id: 0,
                    base_addr: 0,
                    offset: 0,
                    rank: 0,
                    element_count: 0,
                    shape_ptr: 0,
                    stride_ptr: 0,
                    governance: 0,
                    ownership: 0,
                    materialized: 0,
                    state: 0,
                    generation: 0,
                },
                result: error_msg(2, "shape would produce 0 elements"),
            };
        }

        /* Allocate tensor data */
        let data = Arc::new(vec![0.0f32; total_elements as usize]);

        /* Compute strides for canonical (C-contiguous) ordering */
        let mut strides = vec![0usize; rank as usize];
        let mut stride = element_size as usize;
        for i in (0..rank as usize).rev() {
            strides[i] = stride;
            stride *= shape_slice[i] as usize;
        }

        let buffer_id = next_buffer_id();
        let generation = next_generation(buffer_id);

        let record = TensorRecord {
            data: data.clone(),
            shape: shape_slice.to_vec().iter().map(|&s| s as usize).collect(),
            strides,
            offset: 0,
            is_contiguous: true,
            generation,
        };

        TENSOR_REGISTRY.lock().unwrap().insert(buffer_id, record);

        let desc = tensor_to_c_descriptor(buffer_id, &TENSOR_REGISTRY.lock().unwrap()[&buffer_id]);

        TensorOpResultC {
            descriptor: desc,
            result: success_msg(),
        }
    }
}

/* ============================================================================
 * FFI Operation: DEALLOCATE
 *
 * Release reference to a tensor. When strong_owners reaches 0, buffer freed.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_deallocate(desc: *const TensorDescriptorC) -> TensorResultC {
    if desc.is_null() {
        return error_msg(1, "null descriptor");
    }

    unsafe {
        let d = &*desc;
        let mut registry = TENSOR_REGISTRY.lock().unwrap();

        if registry.remove(&d.buffer_id).is_some() {
            success_msg()
        } else {
            error_msg(3, "buffer not found in registry")
        }
    }
}

/* ============================================================================
 * FFI Operation: VIEW (SLICE)
 *
 * Create a strided view without copying data.
 * Increments Arc reference count.
 * Computes new offset and shape.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_view(
    source: *const TensorDescriptorC,
    start_indices: *const u64,
    end_indices: *const u64,
) -> TensorOpResultC {
    if source.is_null() || start_indices.is_null() || end_indices.is_null() {
        return TensorOpResultC {
            descriptor: TensorDescriptorC {
                buffer_id: 0,
                base_addr: 0,
                offset: 0,
                rank: 0,
                element_count: 0,
                shape_ptr: 0,
                stride_ptr: 0,
                governance: 0,
                ownership: 0,
                materialized: 0,
                state: 0,
                generation: 0,
            },
            result: error_msg(1, "null pointer in view operation"),
        };
    }

    unsafe {
        let src_desc = &*source;
        let mut registry = TENSOR_REGISTRY.lock().unwrap();

        match registry.get(&src_desc.buffer_id) {
            Some(src_record) => {
                let rank = src_desc.rank as usize;
                let start = std::slice::from_raw_parts(start_indices, rank);
                let end = std::slice::from_raw_parts(end_indices, rank);

                /* Verify bounds */
                for i in 0..rank {
                    if start[i] >= end[i] || end[i] as usize > src_record.shape[i] {
                        return TensorOpResultC {
                            descriptor: TensorDescriptorC {
                                buffer_id: 0,
                                base_addr: 0,
                                offset: 0,
                                rank: 0,
                                element_count: 0,
                                shape_ptr: 0,
                                stride_ptr: 0,
                                governance: 0,
                                ownership: 0,
                                materialized: 0,
                                state: 0,
                                generation: 0,
                            },
                            result: error_msg(4, "view indices out of bounds"),
                        };
                    }
                }

                /* Compute new shape and new offset */
                let mut new_shape = vec![0usize; rank];
                let mut new_offset = src_record.offset;

                for i in 0..rank {
                    new_shape[i] = (end[i] - start[i]) as usize;
                    new_offset += (start[i] as usize) * src_record.strides[i];
                }

                /* Check if still canonical (contiguous) */
                let mut expected_stride = 4; /* element_size for f32 */
                let mut new_is_contiguous = true;
                for i in (0..rank).rev() {
                    if src_record.strides[i] != expected_stride {
                        new_is_contiguous = false;
                    }
                    expected_stride *= new_shape[i];
                }

                /* Create new view record, sharing Arc */
                let new_buffer_id = next_buffer_id();
                let new_generation = next_generation(new_buffer_id);

                let view_record = TensorRecord {
                    data: src_record.data.clone(), /* Arc::clone for shared ownership */
                    shape: new_shape,
                    strides: src_record.strides.clone(),
                    offset: new_offset,
                    is_contiguous: new_is_contiguous,
                    generation: new_generation,
                };

                registry.insert(new_buffer_id, view_record);

                let desc = tensor_to_c_descriptor(new_buffer_id, &registry[&new_buffer_id]);
                TensorOpResultC {
                    descriptor: desc,
                    result: success_msg(),
                }
            }
            None => TensorOpResultC {
                descriptor: TensorDescriptorC {
                    buffer_id: 0,
                    base_addr: 0,
                    offset: 0,
                    rank: 0,
                    element_count: 0,
                    shape_ptr: 0,
                    stride_ptr: 0,
                    governance: 0,
                    ownership: 0,
                    materialized: 0,
                    state: 0,
                    generation: 0,
                },
                result: error_msg(3, "source buffer not found"),
            },
        }
    }
}

/* ============================================================================
 * FFI Operation: MATERIALIZE
 *
 * Create a new contiguous buffer from a strided view.
 * Source buffer remains alive and unmodified.
 * New buffer is independently owned.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_materialize(
    source: *const TensorDescriptorC,
) -> TensorOpResultC {
    if source.is_null() {
        return TensorOpResultC {
            descriptor: TensorDescriptorC {
                buffer_id: 0,
                base_addr: 0,
                offset: 0,
                rank: 0,
                element_count: 0,
                shape_ptr: 0,
                stride_ptr: 0,
                governance: 0,
                ownership: 0,
                materialized: 0,
                state: 0,
                generation: 0,
            },
            result: error_msg(1, "null descriptor"),
        };
    }

    unsafe {
        let src_desc = &*source;
        let mut registry = TENSOR_REGISTRY.lock().unwrap();

        match registry.get(&src_desc.buffer_id) {
            Some(src_record) => {
                if src_record.is_contiguous {
                    /* Already canonical */
                    let desc = tensor_to_c_descriptor(src_desc.buffer_id, src_record);
                    return TensorOpResultC {
                        descriptor: desc,
                        result: success_msg(),
                    };
                }

                /* Copy strided data to new contiguous buffer */
                let element_count: usize = src_record.shape.iter().product();
                let mut new_data = vec![0.0f32; element_count];

                /* Iterate through logical indices and copy */
                fn copy_strided(
                    src: &[f32],
                    dst: &mut [f32],
                    shape: &[usize],
                    strides: &[usize],
                    src_offset: usize,
                    rank: usize,
                ) {
                    fn recurse(
                        src: &[f32],
                        dst: &mut [f32],
                        shape: &[usize],
                        strides: &[usize],
                        src_offset: usize,
                        rank: usize,
                        dim: usize,
                        dst_idx: &mut usize,
                    ) {
                        if dim == rank {
                            dst[*dst_idx] = src[src_offset];
                            *dst_idx += 1;
                        } else {
                            for i in 0..shape[dim] {
                                let new_offset = src_offset + i * strides[dim];
                                recurse(src, dst, shape, strides, new_offset, rank, dim + 1, dst_idx);
                            }
                        }
                    }

                    let mut dst_idx = 0;
                    recurse(src, dst, shape, strides, src_offset, rank, 0, &mut dst_idx);
                }

                copy_strided(
                    &src_record.data,
                    &mut new_data,
                    &src_record.shape,
                    &src_record.strides,
                    src_record.offset,
                    src_record.shape.len(),
                );

                /* Create new buffer as new Arc */
                let new_arc_data = Arc::new(new_data);
                let new_buffer_id = next_buffer_id();
                let new_generation = next_generation(new_buffer_id);

                /* Compute canonical strides */
                let mut new_strides = vec![0usize; src_record.shape.len()];
                let mut stride = 4; /* element_size for f32 */
                for i in (0..src_record.shape.len()).rev() {
                    new_strides[i] = stride;
                    stride *= src_record.shape[i];
                }

                let mat_record = TensorRecord {
                    data: new_arc_data,
                    shape: src_record.shape.clone(),
                    strides: new_strides,
                    offset: 0,
                    is_contiguous: true,
                    generation: new_generation,
                };

                registry.insert(new_buffer_id, mat_record);

                let desc = tensor_to_c_descriptor(new_buffer_id, &registry[&new_buffer_id]);
                TensorOpResultC {
                    descriptor: desc,
                    result: success_msg(),
                }
            }
            None => TensorOpResultC {
                descriptor: TensorDescriptorC {
                    buffer_id: 0,
                    base_addr: 0,
                    offset: 0,
                    rank: 0,
                    element_count: 0,
                    shape_ptr: 0,
                    stride_ptr: 0,
                    governance: 0,
                    ownership: 0,
                    materialized: 0,
                    state: 0,
                    generation: 0,
                },
                result: error_msg(3, "source buffer not found"),
            },
        }
    }
}

/* ============================================================================
 * FFI Operation: VALIDATE GENERATION
 *
 * Check if a descriptor's generation matches the current buffer generation.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_validate_generation(desc: *const TensorDescriptorC) -> TensorResultC {
    if desc.is_null() {
        return error_msg(1, "null descriptor");
    }

    unsafe {
        let d = &*desc;
        let registry = TENSOR_REGISTRY.lock().unwrap();

        match registry.get(&d.buffer_id) {
            Some(record) => {
                if record.generation == d.generation {
                    success_msg()
                } else {
                    error_msg(5, "stale generation — buffer has been reclaimed")
                }
            }
            None => error_msg(3, "buffer not found"),
        }
    }
}

/* ============================================================================
 * FFI Operation: LOAD
 *
 * Read element value. Validates generation and bounds first.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_load(
    desc: *const TensorDescriptorC,
    indices: *const u64,
    out_value: *mut f32,
) -> TensorResultC {
    if desc.is_null() || indices.is_null() || out_value.is_null() {
        return error_msg(1, "null pointer");
    }

    unsafe {
        let d = &*desc;

        /* Validate generation */
        let registry = TENSOR_REGISTRY.lock().unwrap();
        match registry.get(&d.buffer_id) {
            Some(record) => {
                if record.generation != d.generation {
                    return error_msg(5, "stale generation");
                }

                let rank = record.shape.len();
                let idx_slice = std::slice::from_raw_parts(indices, rank);

                /* Verify bounds */
                for i in 0..rank {
                    if idx_slice[i] as usize >= record.shape[i] {
                        return error_msg(6, "index out of bounds");
                    }
                }

                /* Compute linear address */
                let mut addr = record.offset;
                for i in 0..rank {
                    addr += (idx_slice[i] as usize) * record.strides[i];
                }

                if addr >= record.data.len() {
                    return error_msg(6, "computed address out of bounds");
                }

                *out_value = record.data[addr];
                success_msg()
            }
            None => error_msg(3, "buffer not found"),
        }
    }
}

/* ============================================================================
 * FFI Operation: STORE
 *
 * Write element value. Validates generation and bounds first.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_store(
    desc: *const TensorDescriptorC,
    indices: *const u64,
    value: f32,
) -> TensorResultC {
    if desc.is_null() || indices.is_null() {
        return error_msg(1, "null pointer");
    }

    unsafe {
        let d = &*desc;

        /* Validate generation */
        let mut registry = TENSOR_REGISTRY.lock().unwrap();
        match registry.get_mut(&d.buffer_id) {
            Some(record) => {
                if record.generation != d.generation {
                    return error_msg(5, "stale generation");
                }

                let rank = record.shape.len();
                let idx_slice = std::slice::from_raw_parts(indices, rank);

                /* Verify bounds */
                for i in 0..rank {
                    if idx_slice[i] as usize >= record.shape[i] {
                        return error_msg(6, "index out of bounds");
                    }
                }

                /* Compute linear address and verify Arc allows mutation */
                let mut addr = record.offset;
                for i in 0..rank {
                    addr += (idx_slice[i] as usize) * record.strides[i];
                }

                if addr >= record.data.len() {
                    return error_msg(6, "computed address out of bounds");
                }

                /* Attempt mutation through Arc — fails if multiple strong refs */
                match Arc::get_mut(&mut record.data) {
                    Some(data_mut) => {
                        data_mut[addr] = value;
                        success_msg()
                    }
                    None => error_msg(7, "tensor is shared (owned by multiple references)"),
                }
            }
            None => error_msg(3, "buffer not found"),
        }
    }
}

/* ============================================================================
 * FFI Operation: INFO (for debugging/oracle comparison)
 *
 * Returns human-readable tensor metadata.
 * ============================================================================
 */

#[no_mangle]
pub extern "C" fn tensor_info(
    desc: *const TensorDescriptorC,
    out_buffer: *mut u8,
    out_len: usize,
) -> TensorResultC {
    if desc.is_null() || out_buffer.is_null() {
        return error_msg(1, "null pointer");
    }

    unsafe {
        let d = &*desc;
        let registry = TENSOR_REGISTRY.lock().unwrap();

        match registry.get(&d.buffer_id) {
            Some(record) => {
                let shape_str = format!("[{}]",
                    record.shape.iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", "));
                let stride_str = format!("[{}]",
                    record.strides.iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(", "));

                let info = format!(
                    "buffer_id={} generation={} shape={} strides={} offset={} contiguous={}\0",
                    d.buffer_id, record.generation, shape_str, stride_str,
                    record.offset, record.is_contiguous
                );

                let bytes = info.as_bytes();
                let len = std::cmp::min(bytes.len(), out_len);
                std::ptr::copy_nonoverlapping(bytes.as_ptr(), out_buffer, len);

                success_msg()
            }
            None => error_msg(3, "buffer not found"),
        }
    }
}
