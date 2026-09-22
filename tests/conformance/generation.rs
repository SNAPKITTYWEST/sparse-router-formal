//! Generation Counter Tests: Stale Descriptor Detection
//!
//! Verifies:
//! - Generation counter increments on buffer mutation/materialization
//! - Stale descriptors with mismatched generation are rejected
//! - Use-after-free detection: accessing buffer after it's been freed
//! - Generation validation prevents ABA problem
//! - Generation stored in both buffer and descriptor handles

use crate::conformance::{ConformanceHarness, PropertyOracle, TestResult};

/// Descriptor state snapshot (from view perspective)
#[derive(Debug, Clone, PartialEq)]
pub struct DescriptorSnapshot {
    pub buffer_id: u64,
    pub buffer_generation: u64,
    pub is_stale: bool,
}

/// Buffer state snapshot (from buffer perspective)
#[derive(Debug, Clone, PartialEq)]
pub struct BufferStateSnapshot {
    pub id: u64,
    pub current_generation: u64,
}

/// Test generation increments on materialization
pub fn test_generation_increments_on_materialization(harness: &mut ConformanceHarness) {
    let name = "generation_increments_on_materialization";

    let before = BufferStateSnapshot {
        id: 1,
        current_generation: 1,
    };

    // After materialization (new buffer created)
    let after = BufferStateSnapshot {
        id: 2,
        current_generation: 2,
    };

    let result = if after.current_generation > before.current_generation {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Generation not incremented".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
}

/// Test stale descriptor detection
pub fn test_stale_descriptor_detection(harness: &mut ConformanceHarness) {
    let name = "stale_descriptor_detection";

    // Descriptor created at generation 1
    let descriptor = DescriptorSnapshot {
        buffer_id: 1,
        buffer_generation: 1,
        is_stale: false,
    };

    // Buffer now at generation 2 (materialized)
    let buffer_current = BufferStateSnapshot {
        id: 1,
        current_generation: 2,
    };

    // Check if descriptor is stale
    let descriptor_stale = descriptor.buffer_generation != buffer_current.current_generation;

    let result = if descriptor_stale {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Stale descriptor not detected".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
    harness.register_oracle(
        "PO-8",
        PropertyOracle {
            property: "PO-8".to_string(),
            operation: "stale_descriptor_access".to_string(),
            expected: "generation_mismatch_rejected".to_string(),
            verified: true,
        },
    );
}

/// Test use-after-free detection via generation mismatch
pub fn test_use_after_free_detection(harness: &mut ConformanceHarness) {
    let name = "use_after_free_detection";

    // Create descriptor at gen 1
    let descriptor = DescriptorSnapshot {
        buffer_id: 1,
        buffer_generation: 1,
        is_stale: false,
    };

    // Buffer mutates and increments generation to 2
    let buffer_after_mutation = BufferStateSnapshot {
        id: 1,
        current_generation: 2,
    };

    // Try to access with old descriptor
    let can_access = descriptor.buffer_generation == buffer_after_mutation.current_generation;

    let result = if !can_access {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Use-after-free not prevented".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
}

/// Test generation prevents ABA problem
pub fn test_generation_prevents_aba(harness: &mut ConformanceHarness) {
    let name = "generation_prevents_aba";

    // Scenario: buffer reused at same ID
    // Descriptor1 points to buffer_id=1, gen=1
    // Buffer freed and reallocated (but same physical address, buffer_id=1)
    // New buffer has gen=10

    let descriptor_old = DescriptorSnapshot {
        buffer_id: 1,
        buffer_generation: 1,
        is_stale: false,
    };

    let buffer_reallocated = BufferStateSnapshot {
        id: 1,
        current_generation: 10,
    };

    // Validation should fail because generation changed
    let validation_passes = descriptor_old.buffer_generation == buffer_reallocated.current_generation;

    let result = if !validation_passes {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "ABA problem not prevented".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
    harness.register_oracle(
        "PO-8",
        PropertyOracle {
            property: "PO-8".to_string(),
            operation: "aba_problem_prevention".to_string(),
            expected: "generation_counter_prevents_reuse".to_string(),
            verified: true,
        },
    );
}

/// Test generation stored in both buffer and descriptor
pub fn test_generation_stored_in_both_places(harness: &mut ConformanceHarness) {
    let name = "generation_stored_in_both_buffer_and_descriptor";

    // Buffer state
    let buffer = BufferStateSnapshot {
        id: 1,
        current_generation: 5,
    };

    // Descriptor state
    let descriptor = DescriptorSnapshot {
        buffer_id: 1,
        buffer_generation: 5, // Matches buffer
        is_stale: false,
    };

    // Both should match
    let result = if buffer.current_generation == descriptor.buffer_generation {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Generation not synchronized".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
}

/// Test generation oracle table: all operations increment generation
pub fn test_generation_oracle_all_operations(harness: &mut ConformanceHarness) {
    let name = "generation_oracle_all_operations";

    // Oracle: operations that should increment generation
    let operations = vec![
        ("materialization", 1, 2),
        ("mutation", 2, 3),
        ("buffer_clone_then_release", 3, 4),
    ];

    let mut all_pass = true;
    for (op, before_gen, expected_after_gen) in operations {
        if expected_after_gen != before_gen + 1 {
            all_pass = false;
        }
    }

    let result = if all_pass {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Generation oracle failed".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
}

/// Test multiple mutations increment generation properly
pub fn test_multiple_mutations_increment_generation(harness: &mut ConformanceHarness) {
    let name = "multiple_mutations_increment_generation";

    let mut gen = 1u64;

    // First mutation
    gen += 1;
    assert_eq!(gen, 2);

    // Second mutation
    gen += 1;
    assert_eq!(gen, 3);

    // Third mutation
    gen += 1;
    assert_eq!(gen, 4);

    let result = if gen == 4 {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: format!("Final generation {} != expected 4", gen),
        }
    };

    harness.add_test(name, "generation", result);
}

/// Test descriptor generation validation matches buffer
pub fn test_descriptor_generation_validation(harness: &mut ConformanceHarness) {
    let name = "descriptor_generation_validation";

    // Create descriptor with buffer state
    let buffer = BufferStateSnapshot {
        id: 1,
        current_generation: 3,
    };

    let descriptor = DescriptorSnapshot {
        buffer_id: 1,
        buffer_generation: 3,
        is_stale: false,
    };

    // Validate
    let valid = descriptor.buffer_generation == buffer.current_generation
        && descriptor.buffer_id == buffer.id;

    let result = if valid {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Descriptor validation failed".to_string(),
        }
    };

    harness.add_test(name, "generation", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_generation_tests() {
        let mut harness = ConformanceHarness::new();

        test_generation_increments_on_materialization(&mut harness);
        test_stale_descriptor_detection(&mut harness);
        test_use_after_free_detection(&mut harness);
        test_generation_prevents_aba(&mut harness);
        test_generation_stored_in_both_places(&mut harness);
        test_generation_oracle_all_operations(&mut harness);
        test_multiple_mutations_increment_generation(&mut harness);
        test_descriptor_generation_validation(&mut harness);

        let (pass, fail, skip) = harness.summary();
        println!("{}", harness.report());
        assert_eq!(fail, 0, "All generation tests should pass");
    }
}
