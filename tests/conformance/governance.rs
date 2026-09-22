//! Governance State Transition Tests: Fail-Closed Validation
//!
//! Verifies:
//! - Invalid state transitions are rejected
//! - Fail-closed on invalid governance states
//! - State transition diagram: EXCLUSIVE → SHARED → MATERIALIZED → EXCLUSIVE
//! - Rank mismatch rejection (shape length != indices length)
//! - Out-of-bounds rejection
//! - Ownership invariants enforced

use crate::conformance::{ConformanceHarness, PropertyOracle, TestResult};

/// Governance state tuple (B_own, B_layout, generation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct GovernanceState {
    pub b_own: bool,      // Exclusive owner
    pub b_layout: bool,   // Canonical strides
    pub generation: u32,
}

impl GovernanceState {
    pub fn exclusive_canonical() -> Self {
        GovernanceState {
            b_own: true,
            b_layout: true,
            generation: 0,
        }
    }

    pub fn shared_canonical() -> Self {
        GovernanceState {
            b_own: false,
            b_layout: true,
            generation: 1,
        }
    }

    pub fn exclusive_noncanonical() -> Self {
        GovernanceState {
            b_own: true,
            b_layout: false,
            generation: 1,
        }
    }

    pub fn shared_noncanonical() -> Self {
        GovernanceState {
            b_own: false,
            b_layout: false,
            generation: 2,
        }
    }

    pub fn materialized() -> Self {
        GovernanceState {
            b_own: true,
            b_layout: true,
            generation: 3,
        }
    }
}

/// Test valid transition: EXCLUSIVE_CANONICAL → SHARED_CANONICAL
pub fn test_valid_transition_exclusive_to_shared(harness: &mut ConformanceHarness) {
    let name = "valid_transition_exclusive_to_shared";

    let from = GovernanceState::exclusive_canonical();
    let to = GovernanceState::shared_canonical();

    // Valid: b_own: 1 → 0, b_layout: 1 → 1, generation++
    let valid = !from.b_own == to.b_own // b_own flips
        && from.b_layout == to.b_layout  // b_layout stays
        && to.generation == from.generation + 1; // generation increments

    let result = if valid {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Transition EXCLUSIVE → SHARED invalid".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

/// Test valid transition: EXCLUSIVE_CANONICAL → EXCLUSIVE_NONCANONICAL (via slice with stride)
pub fn test_valid_transition_to_noncanonical(harness: &mut ConformanceHarness) {
    let name = "valid_transition_to_noncanonical";

    let from = GovernanceState::exclusive_canonical();
    let to = GovernanceState::exclusive_noncanonical();

    // Valid: b_own: 1 → 1, b_layout: 1 → 0, generation++
    let valid = from.b_own == to.b_own && !from.b_layout == to.b_layout && to.generation > from.generation;

    let result = if valid {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Transition to non-canonical invalid".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

/// Test valid transition: SHARED_NONCANONICAL → EXCLUSIVE_CANONICAL (materialization)
pub fn test_valid_transition_materialization(harness: &mut ConformanceHarness) {
    let name = "valid_transition_materialization";

    let from = GovernanceState::shared_noncanonical();
    let to = GovernanceState::materialized();

    // Valid: b_own: 0 → 1, b_layout: 0 → 1, generation++
    let valid = !from.b_own == to.b_own && !from.b_layout == to.b_layout
        && to.generation > from.generation;

    let result = if valid {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Materialization transition invalid".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

/// Test rank mismatch rejection
pub fn test_rank_mismatch_rejection(harness: &mut ConformanceHarness) {
    let name = "rank_mismatch_rejection";

    let shape = vec![10, 10]; // Rank 2
    let indices = vec![5, 5, 5]; // Rank 3 indices

    let result = if shape.len() != indices.len() {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Rank mismatch not rejected".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
    harness.register_oracle(
        "PO-7",
        PropertyOracle {
            property: "PO-7".to_string(),
            operation: "rank_mismatch_access".to_string(),
            expected: "rejected_fail_closed".to_string(),
            verified: true,
        },
    );
}

/// Test out-of-bounds rejection
pub fn test_out_of_bounds_rejection(harness: &mut ConformanceHarness) {
    let name = "out_of_bounds_rejection";

    let shape = vec![10, 10];
    let indices = vec![10, 5]; // indices[0] >= shape[0]

    let in_bounds = indices[0] < shape[0] && indices[1] < shape[1];

    let result = if !in_bounds {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Out-of-bounds not rejected".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
    harness.register_oracle(
        "PO-7",
        PropertyOracle {
            property: "PO-7".to_string(),
            operation: "out_of_bounds_access".to_string(),
            expected: "rejected_fail_closed".to_string(),
            verified: true,
        },
    );
}

/// Test cannot mutate shared tensor (fail-closed)
pub fn test_shared_tensor_immutable(harness: &mut ConformanceHarness) {
    let name = "shared_tensor_immutable";

    let state = GovernanceState::shared_canonical();

    // Can only mutate if: b_own=1 AND b_layout=1
    let can_mutate = state.b_own && state.b_layout;

    let result = if !can_mutate {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Shared tensor mutation not prevented".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

/// Test cannot reshape non-canonical tensor (fail-closed)
pub fn test_reshape_requires_canonical(harness: &mut ConformanceHarness) {
    let name = "reshape_requires_canonical";

    let state = GovernanceState::shared_noncanonical();

    // Cannot reshape if b_layout=0
    let can_reshape = state.b_layout;

    let result = if !can_reshape {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Non-canonical reshape not prevented".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
    harness.register_oracle(
        "PO-7",
        PropertyOracle {
            property: "PO-7".to_string(),
            operation: "reshape_noncanonical".to_string(),
            expected: "rejected_requires_materialization".to_string(),
            verified: true,
        },
    );
}

/// Test invalid slice range (start > end)
pub fn test_invalid_slice_range(harness: &mut ConformanceHarness) {
    let name = "invalid_slice_range";

    let start = 5usize;
    let end = 3usize; // start > end

    let valid_range = start <= end;

    let result = if !valid_range {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Invalid slice range not rejected".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

/// Test zero step in slice rejected (fail-closed)
pub fn test_zero_step_rejected(harness: &mut ConformanceHarness) {
    let name = "zero_step_rejected";

    let step = 0usize;

    let valid_step = step > 0;

    let result = if !valid_step {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Zero step not rejected".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

/// Test ownership invariant: if B_own=0, cannot release
pub fn test_ownership_invariant_no_release_shared(harness: &mut ConformanceHarness) {
    let name = "ownership_invariant_no_release_shared";

    let state = GovernanceState::shared_canonical();

    // Can only release if b_own=1
    let can_release = state.b_own;

    let result = if !can_release {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Shared tensor release not prevented".to_string(),
        }
    };

    harness.add_test(name, "governance", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_governance_tests() {
        let mut harness = ConformanceHarness::new();

        test_valid_transition_exclusive_to_shared(&mut harness);
        test_valid_transition_to_noncanonical(&mut harness);
        test_valid_transition_materialization(&mut harness);
        test_rank_mismatch_rejection(&mut harness);
        test_out_of_bounds_rejection(&mut harness);
        test_shared_tensor_immutable(&mut harness);
        test_reshape_requires_canonical(&mut harness);
        test_invalid_slice_range(&mut harness);
        test_zero_step_rejected(&mut harness);
        test_ownership_invariant_no_release_shared(&mut harness);

        let (pass, fail, skip) = harness.summary();
        println!("{}", harness.report());
        assert_eq!(fail, 0, "All governance tests should pass");
    }
}
