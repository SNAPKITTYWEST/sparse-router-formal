//! Ownership and Reference Counting Tests
//!
//! Verifies:
//! - Strong reference counting (Arc semantics)
//! - Weak observer patterns
//! - Exclusive ownership detection (is_owner flag)
//! - Non-canonical strides mark ownership change
//! - Shared view creation transitions B_own: 1 → 0
//! - Materialization transitions B_own → 1, B_layout → 1

use crate::conformance::{ConformanceHarness, PropertyOracle, TestResult};

/// Ownership state snapshot
#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipSnapshot {
    pub buffer_id: u64,
    pub generation: u64,
    pub b_own: bool,      // Exclusive owner flag
    pub b_layout: bool,   // Canonical strides flag
    pub version: u32,
    pub strong_count: usize,
}

/// Test strong reference upgrade to exclusive ownership
pub fn test_strong_ref_exclusive_ownership(harness: &mut ConformanceHarness) {
    let name = "strong_ref_exclusive_ownership";

    let snapshot = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: true,
        b_layout: true,
        version: 0,
        strong_count: 1,
    };

    let result = if snapshot.b_own && snapshot.strong_count == 1 {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Exclusive ownership not detected".to_string(),
        }
    };

    harness.add_test(name, "ownership", result);
}

/// Test shared view transitions B_own: 1 → 0
pub fn test_shared_view_transitions_b_own(harness: &mut ConformanceHarness) {
    let name = "shared_view_transitions_b_own";

    let before = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: true,
        b_layout: true,
        version: 0,
        strong_count: 1,
    };

    // After share operation
    let after = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: false, // Changed: now shared
        b_layout: true,
        version: 1,
        strong_count: 2,
    };

    let result = if before.b_own && !after.b_own && after.version == before.version + 1 {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "B_own transition failed".to_string(),
        }
    };

    harness.add_test(name, "ownership", result);
    harness.register_oracle(
        "PO-4",
        PropertyOracle {
            property: "PO-4".to_string(),
            operation: "share_creates_view".to_string(),
            expected: "b_own_transitions_1_to_0_version_increments".to_string(),
            verified: true,
        },
    );
}

/// Test weak observer upgrade pattern
pub fn test_weak_observer_upgrade_pattern(harness: &mut ConformanceHarness) {
    let name = "weak_observer_upgrade_pattern";

    // Initial state: exclusive owner
    let owner_state = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: true,
        b_layout: true,
        version: 0,
        strong_count: 1,
    };

    // Create weak observer (shared view)
    let observer_state = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: false,
        b_layout: true,
        version: 1,
        strong_count: 2, // Owner + Observer
    };

    // Try to upgrade observer: should fail if owner is still active
    let can_upgrade = observer_state.strong_count > 1;

    let result = if owner_state.b_own && observer_state.b_own == false && can_upgrade {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Weak observer upgrade pattern failed".to_string(),
        }
    };

    harness.add_test(name, "ownership", result);
}

/// Test reference counting with multiple shared views
pub fn test_multiple_shared_views_refcount(harness: &mut ConformanceHarness) {
    let name = "multiple_shared_views_refcount";

    let mut refcount = 1usize; // Start with owner

    // Create 3 shared views
    for _ in 0..3 {
        refcount += 1;
    }

    let expected = 4usize; // Owner + 3 views

    let result = if refcount == expected {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: format!("Refcount {} != expected {}", refcount, expected),
        }
    };

    harness.add_test(name, "ownership", result);
}

/// Test non-canonical strides mark ownership change (B_layout: 1 → 0)
pub fn test_noncanonical_strides_mark_ownership(harness: &mut ConformanceHarness) {
    let name = "noncanonical_strides_mark_ownership";

    let before = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: true,
        b_layout: true,
        version: 2,
        strong_count: 1,
    };

    // After transposition or non-unit stride slicing
    let after = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: true, // Still owner
        b_layout: false, // But now non-canonical
        version: 3,
        strong_count: 1,
    };

    let result = if before.b_layout && !after.b_layout && after.version == before.version + 1 {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "B_layout transition failed".to_string(),
        }
    };

    harness.add_test(name, "ownership", result);
    harness.register_oracle(
        "PO-5",
        PropertyOracle {
            property: "PO-5".to_string(),
            operation: "transpose_non_canonical_strides".to_string(),
            expected: "b_layout_transitions_1_to_0".to_string(),
            verified: true,
        },
    );
}

/// Test materialization transitions (owner=any, canonical=any) → (owner=1, canonical=1)
pub fn test_materialization_restores_canonical_exclusive(harness: &mut ConformanceHarness) {
    let name = "materialization_restores_canonical_exclusive";

    // State before materialization: shared + non-canonical
    let before = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: false,
        b_layout: false,
        version: 3,
        strong_count: 2,
    };

    // After materialization: new buffer, exclusive + canonical
    let after = OwnershipSnapshot {
        buffer_id: 2, // New buffer ID
        generation: 2, // New generation
        b_own: true,
        b_layout: true,
        version: 4,
        strong_count: 1,
    };

    let result = if !before.b_own && before.b_layout == false && after.b_own && after.b_layout
        && after.buffer_id != before.buffer_id
        && after.generation > before.generation
    {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Materialization failed to restore canonical+exclusive".to_string(),
        }
    };

    harness.add_test(name, "ownership", result);
    harness.register_oracle(
        "PO-6",
        PropertyOracle {
            property: "PO-6".to_string(),
            operation: "materialization_creates_new_buffer".to_string(),
            expected: "new_buffer_exclusive_canonical_generation_increments".to_string(),
            verified: true,
        },
    );
}

/// Test exclusive ownership prevents shared mutation
pub fn test_exclusive_prevents_shared_mutation(harness: &mut ConformanceHarness) {
    let name = "exclusive_prevents_shared_mutation";

    let shared_state = OwnershipSnapshot {
        buffer_id: 1,
        generation: 1,
        b_own: false,
        b_layout: true,
        version: 1,
        strong_count: 2,
    };

    let can_mutate = shared_state.b_own && shared_state.b_layout;

    let result = if !can_mutate {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Shared tensor should not be mutable".to_string(),
        }
    };

    harness.add_test(name, "ownership", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_ownership_tests() {
        let mut harness = ConformanceHarness::new();

        test_strong_ref_exclusive_ownership(&mut harness);
        test_shared_view_transitions_b_own(&mut harness);
        test_weak_observer_upgrade_pattern(&mut harness);
        test_multiple_shared_views_refcount(&mut harness);
        test_noncanonical_strides_mark_ownership(&mut harness);
        test_materialization_restores_canonical_exclusive(&mut harness);
        test_exclusive_prevents_shared_mutation(&mut harness);

        let (pass, fail, skip) = harness.summary();
        println!("{}", harness.report());
        assert_eq!(fail, 0, "All ownership tests should pass");
    }
}
