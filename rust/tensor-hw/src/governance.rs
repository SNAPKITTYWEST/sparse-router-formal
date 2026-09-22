//! Governance state engine with validation gates
//!
//! Enforces invariants:
//! - No hidden ownership changes
//! - Explicit state transitions
//! - Generation counter synchronization
//! - Fail-closed validation on all operations

use crate::buffer::Buffer;
use crate::error::{HardwareError, Result};
use crate::view::TensorView;

/// Governance validation context
#[derive(Debug, Clone)]
pub struct GovernanceValidator {
    /// Global operation counter for ordering
    op_sequence: u64,
}

impl GovernanceValidator {
    pub fn new() -> Self {
        GovernanceValidator { op_sequence: 0 }
    }

    /// Validate view against buffer
    ///
    /// **Checks**:
    /// - Buffer exists and generation matches
    /// - Shape and strides are consistent
    /// - Offset is within bounds
    /// - Ownership state is valid
    pub fn validate_view(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        // Check buffer identity and generation
        view.buffer_handle.validate(buffer)?;

        // Check buffer mutability constraints
        if view.ownership.can_mutate_directly() {
            buffer.check_mutable()?;
        }

        // Check offset is reasonable
        if view.offset >= buffer.capacity() {
            return Err(HardwareError::AddressOutOfBounds {
                address: view.offset,
                capacity: buffer.capacity(),
            });
        }

        // Check element count doesn't exceed buffer
        let max_elements = buffer.capacity();
        if view.element_count() > max_elements {
            return Err(HardwareError::GovernanceViolation {
                reason: "View element count exceeds buffer capacity",
            });
        }

        Ok(())
    }

    /// Validate element access
    ///
    /// **Checks**:
    /// - Indices within shape bounds
    /// - Computed address within buffer
    /// - Read allowed (always true)
    /// - Write allowed if exclusive owner
    pub fn validate_access(
        &mut self,
        view: &TensorView,
        buffer: &Buffer,
        indices: &[usize],
        is_write: bool,
    ) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        // First validate view integrity
        self.validate_view(view, buffer)?;

        // Check index dimensions
        if indices.len() != view.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: view.shape.len(),
                actual: indices.len(),
            });
        }

        // Check bounds
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= view.shape[i] {
                return Err(HardwareError::IndexOutOfBounds {
                    index: idx,
                    size: view.shape[i],
                });
            }
        }

        // Calculate address
        let addr = view.address_for(indices)?;

        // Validate address in buffer
        if addr >= buffer.capacity() {
            return Err(HardwareError::AddressOutOfBounds {
                address: addr,
                capacity: buffer.capacity(),
            });
        }

        // Check write permissions
        if is_write && !view.ownership.can_mutate_directly() {
            return Err(HardwareError::SharedTensorMutation);
        }

        Ok(())
    }

    /// Validate slice operation
    ///
    /// **Checks**:
    /// - Ranges are within view bounds
    /// - Step is non-zero
    /// - Resulting view is valid
    pub fn validate_slice(
        &mut self,
        view: &TensorView,
        ranges: &[(usize, usize, usize)],
    ) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        if ranges.len() != view.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: view.shape.len(),
                actual: ranges.len(),
            });
        }

        for (dim, &(start, end, step)) in ranges.iter().enumerate() {
            if step == 0 {
                return Err(HardwareError::ViewCreationFailed {
                    reason: "Slice step cannot be zero",
                });
            }

            if start > end {
                return Err(HardwareError::IndexOutOfBounds {
                    index: start,
                    size: view.shape[dim],
                });
            }

            if end > view.shape[dim] {
                return Err(HardwareError::IndexOutOfBounds {
                    index: end,
                    size: view.shape[dim],
                });
            }
        }

        Ok(())
    }

    /// Validate transpose operation
    ///
    /// **Checks**:
    /// - Permutation is valid reordering
    /// - All dimensions present
    pub fn validate_transpose(
        &mut self,
        view: &TensorView,
        permutation: &[usize],
    ) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        if permutation.len() != view.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: view.shape.len(),
                actual: permutation.len(),
            });
        }

        let mut seen = vec![false; permutation.len()];
        for &p in permutation {
            if p >= permutation.len() {
                return Err(HardwareError::ViewCreationFailed {
                    reason: "Permutation index out of range",
                });
            }
            if seen[p] {
                return Err(HardwareError::ViewCreationFailed {
                    reason: "Permutation has duplicate indices",
                });
            }
            seen[p] = true;
        }

        Ok(())
    }

    /// Validate reshape operation
    ///
    /// **Checks**:
    /// - New shape has same element count
    /// - Tensor is canonical (no non-canonical reshape)
    pub fn validate_reshape(
        &mut self,
        view: &TensorView,
        new_shape: &[usize],
    ) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        if !view.ownership.is_canonical {
            return Err(HardwareError::ViewCreationFailed {
                reason: "Cannot reshape non-canonical tensor",
            });
        }

        let new_count: usize = new_shape.iter().product();
        if new_count != view.element_count() {
            return Err(HardwareError::ViewCreationFailed {
                reason: "Reshape: element count mismatch",
            });
        }

        Ok(())
    }

    /// Validate materialization
    ///
    /// **Checks**:
    /// - Source is non-canonical
    /// - Buffer generation is current
    pub fn validate_materialize(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        buffer.validate_generation(view.buffer_handle.generation)?;

        if view.ownership.is_canonical {
            return Err(HardwareError::MaterializationFailed {
                reason: "Cannot materialize canonical tensor",
            });
        }

        Ok(())
    }

    /// Validate clone operation (Arc::clone)
    ///
    /// **Checks**:
    /// - Buffer exists and has current generation
    pub fn validate_clone(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);
        buffer.validate_generation(view.buffer_handle.generation)?;
        Ok(())
    }

    /// Validate release operation
    ///
    /// **Checks**:
    /// - View is exclusive owner
    /// - No dangling references
    pub fn validate_release(&mut self, view: &TensorView, buffer: &Buffer) -> Result<()> {
        self.op_sequence = self.op_sequence.saturating_add(1);

        if !view.ownership.can_release() {
            return Err(HardwareError::OwnershipViolation {
                reason: "Cannot release non-exclusive view",
            });
        }

        // Check that buffer has exactly 1 strong reference (this view)
        if buffer.strong_count() != 1 {
            return Err(HardwareError::OwnershipViolation {
                reason: "Cannot release: buffer has multiple references",
            });
        }

        Ok(())
    }

    /// Get current operation sequence number
    pub fn sequence(&self) -> u64 {
        self.op_sequence
    }
}

impl Default for GovernanceValidator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::memory::MemoryLayout;

    fn create_test_buffer() -> Buffer {
        Buffer::allocate(1, 100).unwrap()
    }

    fn create_test_view(buf: &Buffer) -> TensorView {
        let layout = MemoryLayout::new(100, 0, vec![10, 10], vec![10, 1]).unwrap();
        TensorView::from_buffer(buf, layout)
    }

    #[test]
    fn test_validate_view_success() {
        let mut validator = GovernanceValidator::new();
        let buf = create_test_buffer();
        let view = create_test_view(&buf);

        assert!(validator.validate_view(&view, &buf).is_ok());
    }

    #[test]
    fn test_validate_access_read_allowed() {
        let mut validator = GovernanceValidator::new();
        let buf = create_test_buffer();
        let view = create_test_view(&buf);

        assert!(validator.validate_access(&view, &buf, &[2, 3], false).is_ok());
    }

    #[test]
    fn test_validate_access_write_exclusive() {
        let mut validator = GovernanceValidator::new();
        let buf = create_test_buffer();
        let view = create_test_view(&buf);

        assert!(validator.validate_access(&view, &buf, &[2, 3], true).is_ok());
    }

    #[test]
    fn test_validate_access_write_shared_fails() {
        let mut validator = GovernanceValidator::new();
        let buf = create_test_buffer();
        let mut view = create_test_view(&buf);
        view.ownership = view.ownership.transition_to_shared().unwrap();

        assert!(validator.validate_access(&view, &buf, &[2, 3], true).is_err());
    }

    #[test]
    fn test_validate_slice_success() {
        let mut validator = GovernanceValidator::new();
        let buf = create_test_buffer();
        let view = create_test_view(&buf);

        assert!(validator.validate_slice(&view, &[(2, 5, 1), (3, 7, 1)]).is_ok());
    }

    #[test]
    fn test_validate_slice_zero_step_fails() {
        let mut validator = GovernanceValidator::new();
        let buf = create_test_buffer();
        let view = create_test_view(&buf);

        assert!(validator.validate_slice(&view, &[(2, 5, 0), (3, 7, 1)]).is_err());
    }

    #[test]
    fn test_sequence_increments() {
        let mut validator = GovernanceValidator::new();
        assert_eq!(validator.sequence(), 0);

        let buf = create_test_buffer();
        let view = create_test_view(&buf);

        let _ = validator.validate_view(&view, &buf);
        assert_eq!(validator.sequence(), 1);

        let _ = validator.validate_view(&view, &buf);
        assert_eq!(validator.sequence(), 2);
    }
}
