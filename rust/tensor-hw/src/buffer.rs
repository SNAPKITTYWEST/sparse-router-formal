// SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
// Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
// CLONE_GATE: sparse-router-formal::rust::tensor-hw::src::buffer.rs
//! Buffer management with generation counters and reference tracking
//!
//! Each buffer has:
//! - Unique 64-bit ID
//! - Generation counter (increments on allocation/materialization)
//! - Shared Arc-based ownership
//! - Reference count for exclusivity checking

use crate::error::{HardwareError, Result};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);
pub(crate) fn fresh_buffer_id() -> Result<BufferId> {
    NEXT_BUFFER_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
        .map_err(|_| HardwareError::GovernanceViolation {
            reason: "Buffer identifiers exhausted",
        })
}

/// Unique 64-bit buffer identifier
pub type BufferId = u64;

/// Generation counter for stale descriptor detection
pub type Generation = u64;

/// Hardware buffer with generation tracking
#[derive(Debug, Clone)]
pub struct Buffer {
    /// Physical storage (shared ownership)
    data: Arc<Vec<f32>>,

    /// Current generation counter
    pub generation: Generation,

    /// Unique buffer ID
    pub id: BufferId,

    /// True if buffer is the canonical owner (used for COW decisions)
    pub is_owner: bool,
}

impl Buffer {
    /// Create new buffer with fresh generation counter
    ///
    /// **Governance**: B_own=1 (exclusive), generation=1
    pub fn allocate(id: BufferId, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(HardwareError::ZeroElementShape);
        }

        let bytes = size
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or(HardwareError::AllocationFailed { size })?;
        let mut storage = Vec::new();
        storage
            .try_reserve_exact(size)
            .map_err(|_| HardwareError::AllocationFailed { size: bytes })?;
        storage.resize(size, 0.0);
        let data = Arc::new(storage);

        Ok(Buffer {
            data,
            generation: 1,
            id,
            is_owner: true,
        })
    }

    /// Strong reference count (number of Arc holders)
    #[inline]
    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.data)
    }

    /// Check if this buffer has exclusive ownership
    ///
    /// Returns true only if strong_count == 1 AND is_owner flag is set
    #[inline]
    pub fn is_exclusive(&self) -> bool {
        self.is_owner && self.strong_count() == 1
    }

    /// Create a shared view (does not copy data)
    ///
    /// **Governance**: B_own → 0, generation unchanged
    pub fn create_view(&self) -> Self {
        Buffer {
            data: Arc::clone(&self.data),
            generation: self.generation,
            id: self.id,
            is_owner: false, // Shared view
        }
    }

    /// Increment generation counter for new materialization
    ///
    /// **State Transition**: (gen, ownership) → (gen+1, exclusive=1)
    pub fn next_generation(&self) -> Generation {
        self.generation.saturating_add(1)
    }

    /// Validate that descriptor generation matches buffer generation
    ///
    /// **Safety**: Rejects use-after-free by stale descriptor
    pub fn validate_generation(&self, descriptor_gen: Generation) -> Result<()> {
        if descriptor_gen != self.generation {
            return Err(HardwareError::StaleGeneration {
                buffer_id: self.id,
                expected: self.generation,
                actual: descriptor_gen,
            });
        }
        Ok(())
    }

    /// Check mutability: exclusive ownership + non-shared
    pub fn check_mutable(&self) -> Result<()> {
        if !self.is_exclusive() {
            return Err(HardwareError::SharedTensorMutation);
        }
        Ok(())
    }

    /// Total buffer capacity in elements
    #[inline]
    pub fn capacity(&self) -> usize {
        self.data.len()
    }

    /// Direct element access (unchecked - use with bounds validation)
    #[inline]
    pub fn get_unchecked(&self, index: usize) -> f32 {
        self.data[index]
    }

    /// Checked element write requiring exclusive mutable storage.
    #[inline]
    pub fn set(&mut self, index: usize, value: f32) -> Result<()> {
        self.check_mutable()?;
        let capacity = self.capacity();
        let data = Arc::get_mut(&mut self.data).ok_or(HardwareError::SharedTensorMutation)?;
        let slot = data
            .get_mut(index)
            .ok_or(HardwareError::AddressOutOfBounds {
                address: index,
                capacity,
            })?;
        *slot = value;
        Ok(())
    }
}

/// Buffer handle with metadata for efficient lookups
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle {
    pub id: BufferId,
    pub generation: Generation,
}

impl BufferHandle {
    /// Create handle from buffer
    pub fn from_buffer(buf: &Buffer) -> Self {
        BufferHandle {
            id: buf.id,
            generation: buf.generation,
        }
    }

    /// Validate handle against current buffer state
    pub fn validate(&self, buffer: &Buffer) -> Result<()> {
        if self.id != buffer.id {
            return Err(HardwareError::BufferNotFound(self.id));
        }
        buffer.validate_generation(self.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_sets_generation() {
        let buf = Buffer::allocate(1, 100).unwrap();
        assert_eq!(buf.generation, 1);
        assert!(buf.is_exclusive());
    }

    #[test]
    fn test_view_shares_data_not_ownership() {
        let buf = Buffer::allocate(1, 100).unwrap();
        let view = buf.create_view();

        assert_eq!(buf.strong_count(), 2);
        assert!(!view.is_owner);
        assert!(buf.is_owner);
        assert!(!view.is_exclusive());
    }

    #[test]
    fn test_generation_validation() {
        let buf = Buffer::allocate(1, 100).unwrap();
        assert!(buf.validate_generation(1).is_ok());
        assert!(buf.validate_generation(2).is_err());
    }

    #[test]
    fn test_zero_size_allocation_fails() {
        assert!(Buffer::allocate(1, 0).is_err());
    }
}
