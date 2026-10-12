// SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
// Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
// CLONE_GATE: sparse-router-formal::rust::tensor-hw::src::materialize.rs
//! Materialization barrier: transactional conversion to contiguous memory
//!
//! Materializes non-canonical strides into contiguous buffer.
//! **Transactional**: read → alloc → copy → install → commit
//! **Invariant**: Source buffer remains valid and unchanged

use crate::buffer::{Buffer, BufferId};
use crate::error::{HardwareError, Result};
use crate::view::TensorView;

/// Materialization result with new buffer and view
#[derive(Debug, Clone)]
pub struct MaterializationResult {
    /// New contiguous buffer
    pub new_buffer: Buffer,

    /// View of new buffer
    pub new_view: TensorView,

    /// Source buffer ID (for reference)
    pub source_buffer_id: BufferId,
}

/// Transactional materialization barrier
#[derive(Debug)]
pub struct MaterializationBarrier {}

impl MaterializationBarrier {
    pub fn new() -> Self {
        MaterializationBarrier {}
    }

    /// Materialize non-canonical tensor to new contiguous buffer
    ///
    /// **Transaction Steps**:
    /// 1. **Read**: Load source view metadata (safe snapshot)
    /// 2. **Alloc**: Create new buffer with same element count
    /// 3. **Copy**: Copy strided data to new contiguous buffer (non-destructive)
    /// 4. **Install**: Create new view with new buffer ID and generation
    /// 5. **Commit**: Return new buffer (source unchanged)
    ///
    /// **Governance Transitions**:
    /// - New buffer: B_own=1 (exclusive), B_layout=1 (canonical)
    /// - Source: unchanged (still valid)
    /// - New generation counter
    ///
    /// **Safety**: Borrows source storage; returns independent destination storage.
    pub fn materialize(
        &mut self,
        source_view: &TensorView,
        source_buffer: &Buffer,
    ) -> Result<MaterializationResult> {
        // Step 1: Validate preconditions
        source_view.buffer_handle.validate(source_buffer)?;
        source_view.validate_metadata()?;
        crate::memory::MemoryLayout::new(
            source_buffer.capacity(),
            source_view.offset,
            source_view.shape.clone(),
            source_view.strides.clone(),
        )?;

        if source_view.ownership.is_canonical {
            // Already canonical, no need to materialize
            return Err(HardwareError::MaterializationFailed {
                reason: "Tensor is already canonical (materialization not needed)",
            });
        }

        // Step 2: Allocate new buffer
        let element_count = source_view.element_count();
        let new_buffer_id = crate::buffer::fresh_buffer_id()?;

        let mut new_buffer = Buffer::allocate(new_buffer_id, element_count).map_err(|_| {
            HardwareError::AllocationFailed {
                size: element_count.saturating_mul(std::mem::size_of::<f32>()),
            }
        })?;

        // Step 3: Copy strided data to new contiguous buffer
        // For each index in the source, copy to linear index in new buffer
        self.copy_strided_to_linear(source_view, source_buffer, &mut new_buffer)?;

        // Step 4: Create new view with canonical strides
        let canonical_strides = {
            let mut s = vec![0; source_view.shape.len()];
            let mut stride = 1;
            for i in (0..source_view.shape.len()).rev() {
                s[i] = stride;
                stride *= source_view.shape[i];
            }
            s
        };

        let new_view = TensorView {
            buffer_handle: crate::buffer::BufferHandle::from_buffer(&new_buffer),
            shape: source_view.shape.clone(),
            strides: canonical_strides,
            offset: 0, // New buffer starts at offset 0
            ownership: crate::ownership::OwnershipToken::new_exclusive(),
        };

        // Step 5: Commit (return new buffer)
        Ok(MaterializationResult {
            new_buffer,
            new_view,
            source_buffer_id: source_view.buffer_handle.id,
        })
    }

    /// Copy strided data from source to linear destination
    ///
    /// Uses address calculation to read from arbitrary strides,
    /// writes linearly to destination.
    fn copy_strided_to_linear(
        &self,
        source_view: &TensorView,
        source_buffer: &Buffer,
        dest_buffer: &mut Buffer,
    ) -> Result<()> {
        let count = crate::memory::checked_element_count(&source_view.shape)?;
        for linear in 0..count {
            let mut remaining = linear;
            let mut indices = vec![0; source_view.shape.len()];
            for dim in (0..indices.len()).rev() {
                indices[dim] = remaining % source_view.shape[dim];
                remaining /= source_view.shape[dim];
            }
            let address = source_view.address_for(&indices)?;
            if address >= source_buffer.capacity() {
                return Err(HardwareError::AddressOutOfBounds {
                    address,
                    capacity: source_buffer.capacity(),
                });
            }
            dest_buffer.set(linear, source_buffer.get_unchecked(address))?;
        }
        Ok(())
    }
}

impl Default for MaterializationBarrier {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::MemoryLayout;

    #[test]
    fn test_materialize_non_canonical() {
        let mut barrier = MaterializationBarrier::new();

        // Create source buffer with non-canonical strides
        let src_buf = Buffer::allocate(1, 100).unwrap();
        let shape = vec![10, 10];
        let strides = vec![5, 1]; // Non-canonical (canonical would be [10, 1])

        let layout = MemoryLayout::new(100, 0, shape, strides).unwrap();
        let mut view = TensorView::from_buffer(&src_buf, layout);

        // Make view non-canonical
        view.ownership = view.ownership.transition_to_non_canonical().unwrap();

        let result = barrier.materialize(&view, &src_buf).unwrap();

        assert_eq!(result.new_buffer.generation, 1);
        assert!(result.new_view.ownership.is_canonical);
        assert!(result.new_view.ownership.is_owner);
    }

    #[test]
    fn test_materialize_canonical_fails() {
        let mut barrier = MaterializationBarrier::new();

        let src_buf = Buffer::allocate(1, 100).unwrap();
        let layout = MemoryLayout::new(100, 0, vec![10, 10], vec![10, 1]).unwrap();
        let view = TensorView::from_buffer(&src_buf, layout);

        let result = barrier.materialize(&view, &src_buf);
        assert!(result.is_err());
    }

    #[test]
    fn test_source_remains_valid() {
        let mut barrier = MaterializationBarrier::new();

        let src_buf = Buffer::allocate(1, 100).unwrap();
        let shape = vec![10, 10];
        let strides = vec![5, 1];

        let layout = MemoryLayout::new(100, 0, shape, strides).unwrap();
        let mut view = TensorView::from_buffer(&src_buf, layout);
        view.ownership = view.ownership.transition_to_non_canonical().unwrap();

        let src_count_before = src_buf.strong_count();
        let _ = barrier.materialize(&view, &src_buf);
        let src_count_after = src_buf.strong_count();

        // Source buffer should have original refcount
        assert_eq!(src_count_before, src_count_after);
    }
}
