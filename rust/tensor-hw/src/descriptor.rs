//! Hardware-safe tensor descriptor with #[repr(C)] for FFI
//!
//! Canonical binary layout for cross-language compatibility (Pascal, C, Rust)
//! All fields are aligned and packed for deterministic serialization.

use crate::buffer::BufferId;
use crate::error::Result;

/// Ownership state codes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OwnershipCode {
    Shared = 0,
    Exclusive = 1,
}

/// Governance state codes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GovernanceCode {
    NonCanonical = 0,
    Canonical = 1,
}

/// Materialization state codes
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaterializationCode {
    NotMaterialized = 0,
    Materialized = 1,
}

/// Hardware-safe tensor descriptor
///
/// **C-Compatible Layout** (for FFI):
/// ```c
/// typedef struct {
///     uint64_t buffer_id;      // Unique buffer identity
///     uint64_t base_addr;      // Physical base address (unused, for future HW)
///     uint64_t offset;         // Offset into buffer
///
///     uint32_t rank;           // Tensor rank (# dimensions)
///     uint64_t element_count;  // Total elements
///
///     uint64_t shape_ptr;      // Pointer to shape array
///     uint64_t stride_ptr;     // Pointer to stride array
///
///     uint8_t ownership;       // SHARED=0 or EXCLUSIVE=1
///     uint8_t governance;      // NON_CANONICAL=0 or CANONICAL=1
///     uint8_t materialization; // NOT=0 or MATERIALIZED=1
///     uint8_t reserved;        // Padding for alignment
///
///     uint64_t generation;     // Generation counter
/// } TensorDescriptor;
/// ```
#[repr(C)]
#[derive(Debug, Clone)]
pub struct TensorDescriptor {
    pub buffer_id: u64,
    pub base_addr: u64,
    pub offset: u64,

    pub rank: u32,
    pub element_count: u64,

    pub shape_ptr: u64,
    pub stride_ptr: u64,

    pub ownership: u8,
    pub governance: u8,
    pub materialization: u8,
    pub reserved: u8,

    pub generation: u64,
}

impl TensorDescriptor {
    /// Create descriptor from components
    pub fn new(
        buffer_id: BufferId,
        offset: u64,
        rank: u32,
        element_count: u64,
        shape_ptr: u64,
        stride_ptr: u64,
        is_exclusive: bool,
        is_canonical: bool,
        is_materialized: bool,
        generation: u64,
    ) -> Self {
        TensorDescriptor {
            buffer_id,
            base_addr: 0, // Reserved for future hardware use
            offset,
            rank,
            element_count,
            shape_ptr,
            stride_ptr,
            ownership: if is_exclusive {
                OwnershipCode::Exclusive as u8
            } else {
                OwnershipCode::Shared as u8
            },
            governance: if is_canonical {
                GovernanceCode::Canonical as u8
            } else {
                GovernanceCode::NonCanonical as u8
            },
            materialization: if is_materialized {
                MaterializationCode::Materialized as u8
            } else {
                MaterializationCode::NotMaterialized as u8
            },
            reserved: 0,
            generation,
        }
    }

    /// Get ownership as enum
    pub fn get_ownership(&self) -> OwnershipCode {
        match self.ownership {
            0 => OwnershipCode::Shared,
            1 => OwnershipCode::Exclusive,
            _ => OwnershipCode::Shared, // Default to shared for unknown
        }
    }

    /// Get governance as enum
    pub fn get_governance(&self) -> GovernanceCode {
        match self.governance {
            0 => GovernanceCode::NonCanonical,
            1 => GovernanceCode::Canonical,
            _ => GovernanceCode::NonCanonical,
        }
    }

    /// Get materialization as enum
    pub fn get_materialization(&self) -> MaterializationCode {
        match self.materialization {
            0 => MaterializationCode::NotMaterialized,
            1 => MaterializationCode::Materialized,
            _ => MaterializationCode::NotMaterialized,
        }
    }

    /// Check if descriptor indicates exclusive canonical buffer
    pub fn is_directly_mutable(&self) -> bool {
        self.get_ownership() == OwnershipCode::Exclusive
            && self.get_governance() == GovernanceCode::Canonical
    }

    /// Validate descriptor consistency
    pub fn validate(&self) -> Result<()> {
        // Rank must be reasonable
        if self.rank == 0 {
            return Err(crate::error::HardwareError::ZeroElementShape);
        }

        // Element count must be non-zero
        if self.element_count == 0 {
            return Err(crate::error::HardwareError::ZeroElementShape);
        }

        // Pointers should not be null for valid descriptor
        if self.shape_ptr == 0 || self.stride_ptr == 0 {
            return Err(crate::error::HardwareError::ViewCreationFailed {
                reason: "Shape or stride pointer is null",
            });
        }

        Ok(())
    }

    /// Create a view descriptor (shared copy)
    pub fn as_view(&self) -> Self {
        TensorDescriptor {
            buffer_id: self.buffer_id,
            base_addr: self.base_addr,
            offset: self.offset,
            rank: self.rank,
            element_count: self.element_count,
            shape_ptr: self.shape_ptr,
            stride_ptr: self.stride_ptr,
            ownership: OwnershipCode::Shared as u8,
            governance: self.governance,
            materialization: self.materialization,
            reserved: 0,
            generation: self.generation,
        }
    }

    /// Get size in bytes for FFI serialization
    pub const fn size_bytes() -> usize {
        std::mem::size_of::<TensorDescriptor>()
    }
}

/// Builder for constructing descriptors safely
pub struct DescriptorBuilder {
    buffer_id: u64,
    offset: u64,
    rank: u32,
    element_count: u64,
    shape_ptr: u64,
    stride_ptr: u64,
    is_exclusive: bool,
    is_canonical: bool,
    is_materialized: bool,
    generation: u64,
}

impl DescriptorBuilder {
    pub fn new(buffer_id: u64) -> Self {
        DescriptorBuilder {
            buffer_id,
            offset: 0,
            rank: 0,
            element_count: 0,
            shape_ptr: 0,
            stride_ptr: 0,
            is_exclusive: true,
            is_canonical: true,
            is_materialized: false,
            generation: 1,
        }
    }

    pub fn offset(mut self, offset: u64) -> Self {
        self.offset = offset;
        self
    }

    pub fn rank(mut self, rank: u32) -> Self {
        self.rank = rank;
        self
    }

    pub fn element_count(mut self, count: u64) -> Self {
        self.element_count = count;
        self
    }

    pub fn shape_ptr(mut self, ptr: u64) -> Self {
        self.shape_ptr = ptr;
        self
    }

    pub fn stride_ptr(mut self, ptr: u64) -> Self {
        self.stride_ptr = ptr;
        self
    }

    pub fn exclusive(mut self, exclusive: bool) -> Self {
        self.is_exclusive = exclusive;
        self
    }

    pub fn canonical(mut self, canonical: bool) -> Self {
        self.is_canonical = canonical;
        self
    }

    pub fn materialized(mut self, materialized: bool) -> Self {
        self.is_materialized = materialized;
        self
    }

    pub fn generation(mut self, gen: u64) -> Self {
        self.generation = gen;
        self
    }

    pub fn build(self) -> Result<TensorDescriptor> {
        let desc = TensorDescriptor::new(
            self.buffer_id,
            self.offset,
            self.rank,
            self.element_count,
            self.shape_ptr,
            self.stride_ptr,
            self.is_exclusive,
            self.is_canonical,
            self.is_materialized,
            self.generation,
        );

        desc.validate()?;
        Ok(desc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_descriptor_c_layout() {
        // Ensure #[repr(C)] gives expected size
        assert_eq!(TensorDescriptor::size_bytes(), 72); // 64-bit aligned
    }

    #[test]
    fn test_descriptor_exclusive_canonical() {
        let desc = TensorDescriptor::new(1, 0, 2, 100, 100, 108, true, true, false, 1);
        assert!(desc.is_directly_mutable());
    }

    #[test]
    fn test_descriptor_shared_not_mutable() {
        let desc = TensorDescriptor::new(1, 0, 2, 100, 100, 108, false, true, false, 1);
        assert!(!desc.is_directly_mutable());
    }

    #[test]
    fn test_descriptor_builder() {
        let desc = DescriptorBuilder::new(1)
            .rank(2)
            .element_count(100)
            .shape_ptr(100)
            .stride_ptr(108)
            .build()
            .unwrap();

        assert_eq!(desc.buffer_id, 1);
        assert_eq!(desc.rank, 2);
        assert!(desc.is_directly_mutable());
    }

    #[test]
    fn test_descriptor_validation_zero_rank() {
        let desc = DescriptorBuilder::new(1)
            .rank(0)
            .shape_ptr(100)
            .stride_ptr(108)
            .build();

        assert!(desc.is_err());
    }

    #[test]
    fn test_as_view_makes_shared() {
        let desc = TensorDescriptor::new(1, 0, 2, 100, 100, 108, true, true, false, 1);
        let view = desc.as_view();
        assert_eq!(view.get_ownership(), OwnershipCode::Shared);
        assert_eq!(view.buffer_id, desc.buffer_id);
    }
}
