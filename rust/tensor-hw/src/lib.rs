//! # Tensor Hardware Implementation Layer
//!
//! NraayTensor hardware layer with explicit state transitions, generation-based safety,
//! and fail-closed validation.
//!
//! ## Modules
//!
//! - **buffer**: Buffer management with generation counters and Arc-based ownership
//! - **ownership**: Ownership token state machine (B_layout, B_own)
//! - **memory**: Memory address calculation and stride handling
//! - **view**: Tensor views and slicing operations (zero-copy)
//! - **materialize**: Materialization barrier (transactional copy-on-write)
//! - **descriptor**: Hardware-safe C-compatible tensor descriptors
//! - **governance**: State validation engine with fail-closed gates
//! - **routing**: Sparse work-stealing with memory locality heuristic
//! - **error**: Fail-closed error types
//!
//! ## Core Operations
//!
//! ```ignore
//! // Allocate tensor
//! let buffer = Buffer::allocate(buffer_id, size)?;
//! let view = TensorView::from_buffer(&buffer, layout);
//!
//! // Create shared view (zero-copy)
//! let shared = view.share()?;
//!
//! // Slice (non-destructive, creates shared view)
//! let sliced = view.slice(&[(0, 5, 1), (0, 5, 1)])?;
//!
//! // Materialize non-canonical to contiguous
//! let result = barrier.materialize(&sliced, &buffer)?;
//! // result.new_buffer is exclusive, canonical, source unchanged
//!
//! // Element access with validation
//! validator.validate_access(&view, &buffer, &[0, 0], false)?;
//! let addr = view.address_for(&[0, 0])?;
//! let value = buffer.get_unchecked(addr);
//! ```
//!
//! ## Governance Model
//!
//! Each tensor has binary semantic governance B = (B_layout, B_own):
//! - **B_layout** (is_canonical): 1 = contiguous with canonical strides, 0 = non-canonical
//! - **B_own** (is_owner): 1 = exclusive owner, 0 = shared view
//!
//! Transitions:
//! - VIEW: (owner, canonical) → (shared, canonical or non-canonical)
//! - SLICE: (owner, canonical) → (shared, non-canonical)
//! - MATERIALIZE: (shared, non-canonical) → (owner, canonical) with new buffer
//!
//! All transitions have generation counters for stale descriptor detection.

#![cfg_attr(not(feature = "std"), no_std)]

pub mod buffer;
pub mod error;
pub mod descriptor;
pub mod governance;
pub mod materialize;
pub mod memory;
pub mod ownership;
pub mod routing;
pub mod view;

pub use buffer::{Buffer, BufferHandle, BufferId};
pub use error::{HardwareError, Result};
pub use descriptor::TensorDescriptor;
pub use governance::GovernanceValidator;
pub use materialize::{MaterializationBarrier, MaterializationResult};
pub use memory::MemoryLayout;
pub use ownership::{OwnershipState, OwnershipToken};
pub use routing::{SparseRouter, Task, RegionId, WorkerId};
pub use view::TensorView;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Core tensor hardware operations
pub struct TensorHardware {
    /// Governance validator
    pub validator: GovernanceValidator,

    /// Materialization barrier
    pub materializer: MaterializationBarrier,

    /// Work-stealing router
    pub router: SparseRouter,

    /// Next buffer ID allocator
    next_buffer_id: u64,
}

impl TensorHardware {
    /// Create new hardware context
    pub fn new() -> Self {
        TensorHardware {
            validator: GovernanceValidator::new(),
            materializer: MaterializationBarrier::new(),
            router: SparseRouter::new(),
            next_buffer_id: 1,
        }
    }

    /// Allocate new tensor buffer
    ///
    /// **Returns**: Exclusive, canonical tensor
    pub fn allocate(&mut self, shape: Vec<usize>) -> Result<(Buffer, TensorView)> {
        if shape.is_empty() {
            return Err(HardwareError::ZeroElementShape);
        }

        let element_count: usize = shape.iter().product();
        if element_count == 0 {
            return Err(HardwareError::ZeroElementShape);
        }

        let buffer_id = self.next_buffer_id;
        self.next_buffer_id += 1;

        let buffer = Buffer::allocate(buffer_id, element_count)?;

        // Create canonical strides
        let mut strides = vec![0; shape.len()];
        let mut stride = 1;
        for i in (0..shape.len()).rev() {
            strides[i] = stride;
            stride *= shape[i];
        }

        let layout = MemoryLayout::new(element_count, 0, shape, strides)?;
        let view = TensorView::from_buffer(&buffer, layout);

        Ok((buffer, view))
    }

    /// Release exclusive tensor (deallocate when refcount reaches 0)
    pub fn release(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()> {
        self.validator.validate_release(view, buffer)?;
        // Arc is automatically dropped when view is dropped
        Ok(())
    }

    /// Create shared view
    pub fn create_view(&mut self, view: &TensorView, buffer: &Buffer) -> Result<TensorView> {
        self.validator.validate_clone(view, buffer)?;
        view.share()
    }

    /// Slice tensor (creates new shared view)
    pub fn slice(
        &mut self,
        view: &TensorView,
        _buffer: &Buffer,
        ranges: &[(usize, usize, usize)],
    ) -> Result<TensorView> {
        self.validator.validate_slice(view, ranges)?;
        view.slice(ranges)
    }

    /// Materialize non-canonical tensor
    pub fn materialize(
        &mut self,
        view: &TensorView,
        buffer: &Buffer,
    ) -> Result<MaterializationResult> {
        self.validator.validate_materialize(view, buffer)?;
        self.materializer.materialize(view, buffer)
    }

    /// Transpose tensor dimensions
    pub fn transpose(
        &mut self,
        view: &TensorView,
        _buffer: &Buffer,
        permutation: &[usize],
    ) -> Result<TensorView> {
        self.validator.validate_transpose(view, permutation)?;
        view.transpose(permutation)
    }

    /// Reshape tensor
    pub fn reshape(
        &mut self,
        view: &TensorView,
        _buffer: &Buffer,
        new_shape: Vec<usize>,
    ) -> Result<TensorView> {
        self.validator.validate_reshape(view, &new_shape)?;
        view.reshape(new_shape)
    }

    /// Get element (read-only)
    pub fn get(
        &mut self,
        view: &TensorView,
        buffer: &Buffer,
        indices: &[usize],
    ) -> Result<f32> {
        self.validator.validate_access(view, buffer, indices, false)?;
        let addr = view.address_for(indices)?;
        Ok(buffer.get_unchecked(addr))
    }

    /// Set element (requires exclusive ownership)
    pub fn set(
        &mut self,
        view: &TensorView,
        buffer: &Buffer,
        indices: &[usize],
        value: f32,
    ) -> Result<()> {
        self.validator.validate_access(view, buffer, indices, true)?;
        let addr = view.address_for(indices)?;
        buffer.set_unchecked(addr, value);
        Ok(())
    }
}

impl Default for TensorHardware {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_access() {
        let mut hw = TensorHardware::new();
        let (buffer, view) = hw.allocate(vec![10, 10]).unwrap();

        assert!(buffer.is_exclusive());
        assert_eq!(view.element_count(), 100);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_exclusive_set_get() {
        let mut hw = TensorHardware::new();
        let (buffer, view) = hw.allocate(vec![10, 10]).unwrap();

        hw.set(&view, &buffer, &[2, 3], 42.0).unwrap();
        let val = hw.get(&view, &buffer, &[2, 3]).unwrap();
        assert_eq!(val, 42.0);
    }

    #[test]
    fn test_shared_cannot_mutate() {
        let mut hw = TensorHardware::new();
        let (buffer, view) = hw.allocate(vec![10, 10]).unwrap();
        let shared = hw.create_view(&view, &buffer).unwrap();

        let result = hw.set(&shared, &buffer, &[2, 3], 42.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_slice_and_transpose() {
        let mut hw = TensorHardware::new();
        let (buffer, view) = hw.allocate(vec![3, 4, 5]).unwrap();

        let sliced = hw.slice(&view, &buffer, &[(0, 2, 1), (0, 3, 1), (0, 4, 1)]).unwrap();
        assert_eq!(sliced.shape, vec![2, 3, 4]);

        let transposed = hw.transpose(&sliced, &buffer, &[2, 0, 1]).unwrap();
        assert_eq!(transposed.shape, vec![4, 2, 3]);
    }

    #[test]
    fn test_materialization_workflow() {
        let mut hw = TensorHardware::new();
        let (buffer, view) = hw.allocate(vec![10, 10]).unwrap();

        // Transpose makes non-canonical
        let transposed = hw.transpose(&view, &buffer, &[1, 0]).unwrap();
        assert!(!transposed.is_contiguous());

        // Materialize creates new contiguous buffer
        let mat_result = hw.materialize(&transposed, &buffer).unwrap();
        assert!(mat_result.new_view.is_contiguous());
        assert!(mat_result.new_view.is_exclusive());
    }
}
