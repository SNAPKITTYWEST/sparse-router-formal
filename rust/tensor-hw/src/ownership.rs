//! Ownership token state machine with explicit transitions
//!
//! Tracks the semantic governance state B = (B_layout, B_own):
//! - B_layout: Is data contiguous with canonical strides?
//! - B_own: Is this view the exclusive owner?
//!
//! State transitions:
//! - OWNED + VIEW → SHARED (B_own: 1 → 0)
//! - SHARED + MATERIALIZE → EXCLUSIVE (new buffer, B_own: 1)
//! - EXCLUSIVE → SHARED (via Arc::clone)

use crate::error::{HardwareError, Result};

/// Ownership and layout governance flags
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OwnershipToken {
    /// B_own: 1 = exclusive owner, 0 = shared view
    pub is_owner: bool,

    /// B_layout: 1 = canonical strides, 0 = non-canonical (needs materialization)
    pub is_canonical: bool,

    /// State version (increments on transition)
    pub version: u32,
}

impl OwnershipToken {
    /// Create initial ownership token (exclusive, canonical)
    ///
    /// **Initial State**: B_own=1, B_layout=1, version=0
    pub fn new_exclusive() -> Self {
        OwnershipToken {
            is_owner: true,
            is_canonical: true,
            version: 0,
        }
    }

    /// Transition to shared view via Arc::clone
    ///
    /// **Transition**: (owner=1) → (owner=0), version++
    pub fn transition_to_shared(&self) -> Result<Self> {
        if !self.is_owner {
            // Already shared, this is a no-op for consistency
            return Ok(OwnershipToken {
                is_owner: false,
                is_canonical: self.is_canonical,
                version: self.version.saturating_add(1),
            });
        }

        Ok(OwnershipToken {
            is_owner: false,
            is_canonical: self.is_canonical,
            version: self.version.saturating_add(1),
        })
    }

    /// Transition to non-canonical strides (via slice)
    ///
    /// **Transition**: (canonical=1) → (canonical=0), version++
    pub fn transition_to_non_canonical(&self) -> Result<Self> {
        Ok(OwnershipToken {
            is_owner: self.is_owner,
            is_canonical: false,
            version: self.version.saturating_add(1),
        })
    }

    /// Transition to exclusive with new generation (materialization)
    ///
    /// **Transition**: (owner=any, canonical=any) → (owner=1, canonical=1), version++
    pub fn transition_to_materialized(&self) -> Result<Self> {
        Ok(OwnershipToken {
            is_owner: true,
            is_canonical: true,
            version: self.version.saturating_add(1),
        })
    }

    /// Check if tensor can be directly mutated
    ///
    /// Only true if: is_owner=1 AND is_canonical=1 (exclusive, contiguous)
    pub fn can_mutate_directly(&self) -> bool {
        self.is_owner && self.is_canonical
    }

    /// Check if materialization is needed
    ///
    /// Returns true if: is_canonical=0 (non-contiguous strides)
    pub fn needs_materialization(&self) -> bool {
        !self.is_canonical
    }

    /// Check if release is safe (exclusive owner)
    ///
    /// Safe to release only if: is_owner=1
    pub fn can_release(&self) -> bool {
        self.is_owner
    }
}

/// Ownership state machine with buffer generation tracking
#[derive(Debug, Clone)]
pub struct OwnershipState {
    token: OwnershipToken,
    /// Associated buffer generation for consistency checks
    buffer_generation: u64,
}

impl OwnershipState {
    /// Create new ownership state
    pub fn new(buffer_generation: u64) -> Self {
        OwnershipState {
            token: OwnershipToken::new_exclusive(),
            buffer_generation,
        }
    }

    /// Get current token
    pub fn token(&self) -> OwnershipToken {
        self.token
    }

    /// Validate against buffer generation
    ///
    /// Ensures state is synchronized with buffer
    pub fn validate(&self, current_generation: u64) -> Result<()> {
        if self.buffer_generation != current_generation {
            return Err(HardwareError::GovernanceViolation {
                reason: "Ownership state out of sync with buffer generation",
            });
        }
        Ok(())
    }

    /// Apply shared transition
    pub fn make_shared(&mut self) -> Result<()> {
        self.token = self.token.transition_to_shared()?;
        Ok(())
    }

    /// Apply non-canonical transition
    pub fn make_non_canonical(&mut self) -> Result<()> {
        self.token = self.token.transition_to_non_canonical()?;
        Ok(())
    }

    /// Apply materialization transition
    pub fn make_materialized(&mut self, new_generation: u64) -> Result<()> {
        self.token = self.token.transition_to_materialized()?;
        self.buffer_generation = new_generation;
        Ok(())
    }

    /// Check release preconditions
    pub fn check_releasable(&self) -> Result<()> {
        if !self.token.can_release() {
            return Err(HardwareError::OwnershipViolation {
                reason: "Cannot release shared tensor (not owner)",
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exclusive_initial_state() {
        let token = OwnershipToken::new_exclusive();
        assert!(token.is_owner);
        assert!(token.is_canonical);
        assert!(token.can_mutate_directly());
    }

    #[test]
    fn test_transition_to_shared() {
        let token = OwnershipToken::new_exclusive();
        let shared = token.transition_to_shared().unwrap();
        assert!(!shared.is_owner);
        assert_eq!(shared.version, 1);
        assert!(!shared.can_mutate_directly());
    }

    #[test]
    fn test_transition_to_non_canonical() {
        let token = OwnershipToken::new_exclusive();
        let non_canon = token.transition_to_non_canonical().unwrap();
        assert!(non_canon.is_owner);
        assert!(!non_canon.is_canonical);
        assert!(non_canon.needs_materialization());
    }

    #[test]
    fn test_materialization_restores_canonical() {
        let token = OwnershipToken::new_exclusive()
            .transition_to_non_canonical()
            .unwrap();
        let materialized = token.transition_to_materialized().unwrap();
        assert!(materialized.is_owner);
        assert!(materialized.is_canonical);
        assert!(materialized.can_mutate_directly());
    }

    #[test]
    fn test_ownership_state_validation() {
        let mut state = OwnershipState::new(1);
        assert!(state.validate(1).is_ok());
        assert!(state.validate(2).is_err());
    }

    #[test]
    fn test_state_machine_sequence() {
        let mut state = OwnershipState::new(1);

        // Exclusive → Shared
        state.make_shared().unwrap();
        assert!(!state.token().is_owner);

        // Still shared after another transition attempt
        state.make_shared().unwrap();
        assert!(!state.token().is_owner);

        // Make non-canonical
        state.make_non_canonical().unwrap();
        assert!(!state.token().is_canonical);

        // Materialize restores canonical + exclusive
        state.make_materialized(2).unwrap();
        assert!(state.token().is_owner);
        assert!(state.token().is_canonical);
    }
}
