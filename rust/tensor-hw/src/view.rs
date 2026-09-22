//! Tensor views and slicing operations (zero-copy)
//!
//! Views share underlying buffer via Arc, never copy data.
//! Slicing creates new semantic shape/strides without modifying source.

use crate::buffer::{Buffer, BufferHandle};
use crate::error::{HardwareError, Result};
use crate::memory::MemoryLayout;
use crate::ownership::OwnershipToken;

/// Tensor view descriptor with semantic shape/strides
#[derive(Clone, Debug)]
pub struct TensorView {
    /// Shared buffer handle
    pub buffer_handle: BufferHandle,

    /// Semantic shape (may be transposed/reshaped)
    pub shape: Vec<usize>,

    /// Memory strides for each dimension
    pub strides: Vec<usize>,

    /// Starting offset in buffer
    pub offset: usize,

    /// Ownership state
    pub ownership: OwnershipToken,
}

impl TensorView {
    /// Create a view from buffer and layout
    pub fn from_buffer(buffer: &Buffer, layout: MemoryLayout) -> Self {
        TensorView {
            buffer_handle: BufferHandle::from_buffer(buffer),
            shape: layout.shape,
            strides: layout.strides,
            offset: layout.offset,
            ownership: OwnershipToken::new_exclusive(),
        }
    }

    /// Create a shared view (Arc::clone)
    ///
    /// **Governance**: B_own: 1 → 0, data NOT copied
    pub fn share(&self) -> Result<Self> {
        let new_ownership = self.ownership.transition_to_shared()?;

        Ok(TensorView {
            buffer_handle: self.buffer_handle,
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: self.offset,
            ownership: new_ownership,
        })
    }

    /// Create a slice view (subset of dimensions)
    ///
    /// **Input**: ranges = [(start, end, step), ...] per dimension
    /// **Result**:
    /// - New shape: (end - start) / step per dimension
    /// - New strides: stride[i] * step
    /// - New offset: offset + Σ(start[i] * stride[i])
    /// - B_own: → 0 (shared)
    /// - B_layout: recomputed
    ///
    /// **Invariant**: Source buffer remains valid and unchanged
    pub fn slice(&self, ranges: &[(usize, usize, usize)]) -> Result<Self> {
        if ranges.len() != self.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: self.shape.len(),
                actual: ranges.len(),
            });
        }

        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_strides = Vec::with_capacity(self.strides.len());
        let mut new_offset = self.offset;

        for (dim, &(start, end, step)) in ranges.iter().enumerate() {
            // Bounds check
            if start > end || end > self.shape[dim] || step == 0 {
                return Err(HardwareError::IndexOutOfBounds {
                    index: start,
                    size: self.shape[dim],
                });
            }

            let new_size = if end > start {
                (end - start + step - 1) / step
            } else {
                0
            };
            new_shape.push(new_size);

            // Update offset
            new_offset = new_offset.saturating_add(start.saturating_mul(self.strides[dim]));

            // Update stride
            new_strides.push(self.strides[dim].saturating_mul(step));
        }

        let new_ownership = self.ownership.transition_to_shared()?;

        Ok(TensorView {
            buffer_handle: self.buffer_handle,
            shape: new_shape,
            strides: new_strides,
            offset: new_offset,
            ownership: new_ownership,
        })
    }

    /// Transpose (permute) dimensions
    ///
    /// **Input**: permutation = new ordering of dimension indices
    /// **Example**: [2, 0, 1] for 3D tensor means new_shape[0] = old_shape[2]
    pub fn transpose(&self, permutation: &[usize]) -> Result<Self> {
        if permutation.len() != self.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: self.shape.len(),
                actual: permutation.len(),
            });
        }

        // Validate permutation is a valid reordering
        let mut seen = vec![false; permutation.len()];
        for &p in permutation {
            if p >= permutation.len() || seen[p] {
                return Err(HardwareError::ViewCreationFailed {
                    reason: "Invalid permutation (not a valid reordering)",
                });
            }
            seen[p] = true;
        }

        // Apply permutation to shape and strides
        let mut new_shape = Vec::with_capacity(self.shape.len());
        let mut new_strides = Vec::with_capacity(self.strides.len());

        for &p in permutation {
            new_shape.push(self.shape[p]);
            new_strides.push(self.strides[p]);
        }

        // Transposition makes strides non-canonical
        let new_ownership = self.ownership.transition_to_non_canonical()?;

        Ok(TensorView {
            buffer_handle: self.buffer_handle,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            ownership: new_ownership,
        })
    }

    /// Reshape to new shape
    ///
    /// **Precondition**: element_count must match
    /// **Note**: May require materialization if strides are non-canonical
    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Self> {
        if new_shape.is_empty() {
            return Err(HardwareError::ZeroElementShape);
        }

        let new_element_count: usize = new_shape.iter().product();
        if new_element_count != self.element_count() {
            return Err(HardwareError::ViewCreationFailed {
                reason: "Reshape: element count mismatch",
            });
        }

        // Can only reshape directly if canonical strides
        if !self.ownership.is_canonical {
            return Err(HardwareError::ViewCreationFailed {
                reason: "Cannot reshape non-canonical tensor (needs materialization first)",
            });
        }

        // Compute new canonical strides
        let mut new_strides = vec![0; new_shape.len()];
        let mut stride: usize = 1;
        for i in (0..new_shape.len()).rev() {
            new_strides[i] = stride;
            stride = stride.saturating_mul(new_shape[i]);
        }

        Ok(TensorView {
            buffer_handle: self.buffer_handle,
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
            ownership: self.ownership,
        })
    }

    /// Get total number of elements
    #[inline]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate address for multi-dimensional index
    pub fn address_for(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: self.shape.len(),
                actual: indices.len(),
            });
        }

        // Bounds check
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape[i] {
                return Err(HardwareError::IndexOutOfBounds {
                    index: idx,
                    size: self.shape[i],
                });
            }
        }

        // Calculate strided address
        let mut addr = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            addr = addr.saturating_add(idx.saturating_mul(self.strides[i]));
        }

        Ok(addr)
    }

    /// Check if this view has canonical strides
    pub fn is_contiguous(&self) -> bool {
        self.ownership.is_canonical
    }

    /// Check if this view is exclusive owner
    pub fn is_exclusive(&self) -> bool {
        self.ownership.is_owner
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_view(shape: Vec<usize>) -> TensorView {
        let buf = Buffer::allocate(1, shape.iter().product()).unwrap();
        let strides = {
            let mut s = vec![0; shape.len()];
            let mut stride = 1;
            for i in (0..shape.len()).rev() {
                s[i] = stride;
                stride *= shape[i];
            }
            s
        };
        let layout = MemoryLayout::new(buf.capacity(), 0, shape, strides).unwrap();
        TensorView::from_buffer(&buf, layout)
    }

    #[test]
    fn test_share_makes_non_exclusive() {
        let view = create_test_view(vec![10, 10]);
        assert!(view.is_exclusive());

        let shared = view.share().unwrap();
        assert!(!shared.is_exclusive());
        assert_eq!(shared.element_count(), 100);
    }

    #[test]
    fn test_slice_reduces_dimensions() {
        let view = create_test_view(vec![10, 10]);
        let sliced = view.slice(&[(2, 5, 1), (3, 7, 1)]).unwrap();
        assert_eq!(sliced.shape, vec![3, 4]);
    }

    #[test]
    fn test_transpose_permutes_shape() {
        let view = create_test_view(vec![2, 3, 4]);
        let transposed = view.transpose(&[2, 0, 1]).unwrap();
        assert_eq!(transposed.shape, vec![4, 2, 3]);
    }

    #[test]
    fn test_reshape_requires_canonical() {
        let view = create_test_view(vec![10, 10]);
        let reshaped = view.reshape(vec![100]).unwrap();
        assert_eq!(reshaped.element_count(), 100);
    }

    #[test]
    fn test_address_calculation() {
        let view = create_test_view(vec![10, 10]);
        let addr = view.address_for(&[2, 3]).unwrap();
        assert_eq!(addr, 2 * 10 + 3);
    }
}
