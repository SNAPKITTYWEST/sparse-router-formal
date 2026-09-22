//! Error types for hardware tensor layer

use thiserror::Error;

/// Hardware tensor operation result type
pub type Result<T> = std::result::Result<T, HardwareError>;

/// Fail-closed errors for tensor hardware operations
#[derive(Error, Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardwareError {
    /// Rank mismatch (expected, actual)
    #[error("Rank mismatch: expected {expected}, got {actual}")]
    RankMismatch { expected: usize, actual: usize },

    /// Shape contains zero dimension
    #[error("Shape contains zero element dimension")]
    ZeroElementShape,

    /// Buffer not found in registry
    #[error("Buffer {0} not found in registry")]
    BufferNotFound(u64),

    /// Generation counter mismatch (use-after-free detection)
    #[error("Stale generation counter for buffer {buffer_id}: expected {expected}, got {actual}")]
    StaleGeneration { buffer_id: u64, expected: u64, actual: u64 },

    /// Index out of bounds
    #[error("Index out of bounds: [{index}] exceeds dimension size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    /// Computed memory address out of tensor bounds
    #[error("Memory address {address} exceeds buffer capacity {capacity}")]
    AddressOutOfBounds { address: usize, capacity: usize },

    /// Shared tensor cannot be mutated
    #[error("Cannot mutate shared tensor (refcount > 1)")]
    SharedTensorMutation,

    /// Invalid stride configuration
    #[error("Invalid stride: cannot compute stride for dimension {dim}")]
    InvalidStride { dim: usize },

    /// Materialization failed
    #[error("Materialization failed: {reason}")]
    MaterializationFailed { reason: &'static str },

    /// Governance state violation
    #[error("Governance violation: {reason}")]
    GovernanceViolation { reason: &'static str },

    /// Ownership invariant violated
    #[error("Ownership invariant violated: {reason}")]
    OwnershipViolation { reason: &'static str },

    /// View creation failed
    #[error("View creation failed: {reason}")]
    ViewCreationFailed { reason: &'static str },

    /// Allocation failed
    #[error("Memory allocation failed for {size} bytes")]
    AllocationFailed { size: usize },
}
