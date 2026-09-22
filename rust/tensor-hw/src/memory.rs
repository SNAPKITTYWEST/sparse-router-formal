//! Memory address calculation and stride handling
//!
//! Core address formula:
//!   addr = base + offset + Σ(index[i] * stride[i])

use crate::error::{HardwareError, Result};

/// Memory domain for address allocation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryDomain {
    /// Standard heap allocation
    Heap,
    /// GPU/accelerator memory (future)
    Accelerator,
    /// Pinned memory for DMA
    Pinned,
}

/// Address calculation context
#[derive(Debug, Clone)]
pub struct MemoryLayout {
    /// Physical buffer capacity
    pub capacity: usize,

    /// Base offset for this tensor in buffer
    pub offset: usize,

    /// Shape dimensions
    pub shape: Vec<usize>,

    /// Stride for each dimension
    pub strides: Vec<usize>,

    /// Memory domain
    pub domain: MemoryDomain,
}

impl MemoryLayout {
    /// Create new memory layout
    ///
    /// **Validation**: offset must be in bounds, shape must not exceed capacity
    pub fn new(
        capacity: usize,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Result<Self> {
        if shape.is_empty() {
            return Err(HardwareError::ZeroElementShape);
        }

        if strides.len() != shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: shape.len(),
                actual: strides.len(),
            });
        }

        // Check if any dimension is zero
        if shape.iter().any(|&s| s == 0) {
            return Err(HardwareError::ZeroElementShape);
        }

        // Validate offset is within capacity
        if offset >= capacity {
            return Err(HardwareError::AddressOutOfBounds {
                address: offset,
                capacity,
            });
        }

        let layout = MemoryLayout {
            capacity,
            offset,
            shape,
            strides,
            domain: MemoryDomain::Heap,
        };

        Ok(layout)
    }

    /// Calculate total number of elements
    #[inline]
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// Calculate address for multi-dimensional index
    ///
    /// **Formula**: addr = offset + Σ(index[i] * stride[i])
    pub fn address_for(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.shape.len() {
            return Err(HardwareError::RankMismatch {
                expected: self.shape.len(),
                actual: indices.len(),
            });
        }

        // Bounds check each index
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

        // Validate address is within capacity
        if addr >= self.capacity {
            return Err(HardwareError::AddressOutOfBounds {
                address: addr,
                capacity: self.capacity,
            });
        }

        Ok(addr)
    }

    /// Check if strides are canonical (row-major)
    ///
    /// Canonical: stride[i] = ∏_{j>i} shape[j]
    pub fn is_canonical_strides(&self) -> bool {
        if self.shape.is_empty() {
            return false;
        }

        let mut expected_stride = 1;
        for i in (0..self.shape.len()).rev() {
            if self.strides[i] != expected_stride {
                return false;
            }
            expected_stride = expected_stride.saturating_mul(self.shape[i]);
        }
        true
    }

    /// Compute canonical strides for current shape
    pub fn canonical_strides(&self) -> Vec<usize> {
        let mut strides = vec![0; self.shape.len()];
        let mut stride: usize = 1;
        for i in (0..self.shape.len()).rev() {
            strides[i] = stride;
            stride = stride.saturating_mul(self.shape[i]);
        }
        strides
    }

    /// Create a view (slice) of this layout
    ///
    /// **Arguments**: ranges = [(start, end, step), ...]
    /// **Returns**: new layout with updated shape, strides, offset
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
            // Validate range
            if start > end || end > self.shape[dim] || step == 0 {
                return Err(HardwareError::IndexOutOfBounds {
                    index: start,
                    size: self.shape[dim],
                });
            }

            let new_size = (end - start + step - 1) / step; // Ceiling division
            new_shape.push(new_size);

            // Update offset for this dimension
            new_offset = new_offset.saturating_add(start.saturating_mul(self.strides[dim]));

            // Update stride (multiply by step)
            new_strides.push(self.strides[dim].saturating_mul(step));
        }

        MemoryLayout {
            capacity: self.capacity,
            offset: new_offset,
            shape: new_shape,
            strides: new_strides,
            domain: self.domain,
        }
        .validate_bounds()
    }

    /// Validate that maximum address is within capacity
    fn validate_bounds(&self) -> Result<Self> {
        if self.element_count() == 0 {
            return Err(HardwareError::ZeroElementShape);
        }

        // Calculate worst-case address (last element)
        let mut max_indices = vec![0; self.shape.len()];
        for (i, &s) in self.shape.iter().enumerate() {
            if s > 0 {
                max_indices[i] = s - 1;
            }
        }

        let _ = self.address_for(&max_indices)?;
        Ok(self.clone())
    }
}

/// Flat (1D) linear address indexing
#[derive(Debug, Clone)]
pub struct LinearIndex {
    pub capacity: usize,
    pub offset: usize,
}

impl LinearIndex {
    pub fn new(capacity: usize, offset: usize) -> Result<Self> {
        if offset >= capacity {
            return Err(HardwareError::AddressOutOfBounds {
                address: offset,
                capacity,
            });
        }
        Ok(LinearIndex { capacity, offset })
    }

    /// Get address for linear index
    #[inline]
    pub fn address(&self, index: usize) -> Result<usize> {
        let addr = self.offset.saturating_add(index);
        if addr >= self.capacity {
            return Err(HardwareError::AddressOutOfBounds {
                address: addr,
                capacity: self.capacity,
            });
        }
        Ok(addr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_layout_creation() {
        let layout = MemoryLayout::new(1000, 0, vec![10, 10], vec![10, 1]).unwrap();
        assert_eq!(layout.element_count(), 100);
    }

    #[test]
    fn test_address_calculation() {
        let layout = MemoryLayout::new(1000, 0, vec![10, 10], vec![10, 1]).unwrap();
        let addr = layout.address_for(&[2, 3]).unwrap();
        assert_eq!(addr, 2 * 10 + 3 * 1); // 23
    }

    #[test]
    fn test_canonical_strides_check() {
        let layout = MemoryLayout::new(1000, 0, vec![10, 10], vec![10, 1]).unwrap();
        assert!(layout.is_canonical_strides());

        let layout2 = MemoryLayout::new(1000, 0, vec![10, 10], vec![5, 1]).unwrap();
        assert!(!layout2.is_canonical_strides());
    }

    #[test]
    fn test_bounds_checking() {
        let layout = MemoryLayout::new(100, 0, vec![10, 10], vec![10, 1]).unwrap();
        assert!(layout.address_for(&[5, 5]).is_ok());
        assert!(layout.address_for(&[10, 5]).is_err());
    }

    #[test]
    fn test_slice_operation() {
        let layout = MemoryLayout::new(1000, 0, vec![10, 10], vec![10, 1]).unwrap();
        let sliced = layout.slice(&[(2, 5, 1), (3, 7, 1)]).unwrap();
        assert_eq!(sliced.shape, vec![3, 4]);
    }

    #[test]
    fn test_zero_element_shape_rejected() {
        assert!(MemoryLayout::new(1000, 0, vec![10, 0], vec![10, 1]).is_err());
    }
}
