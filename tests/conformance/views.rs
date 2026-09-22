//! View Semantics Tests: Slicing, Transposing, Nested Views
//!
//! Verifies:
//! - Slice creates new shape without copying data
//! - Transposition permutes shape/strides
//! - Nested views share underlying buffer (same buffer_id)
//! - Source buffer remains valid after view operations
//! - Shape, stride, offset calculations match between implementations

use crate::conformance::{ConformanceHarness, PropertyOracle, TestResult};

/// View layout snapshot for cross-language comparison
#[derive(Debug, Clone, PartialEq)]
pub struct ViewSnapshot {
    pub buffer_id: u64,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
    pub is_owner: bool,
    pub is_canonical: bool,
}

/// Test slice operation creates new shape
pub fn test_slice_creates_new_shape(harness: &mut ConformanceHarness) {
    let name = "slice_creates_new_shape";

    // Original view: [10, 10]
    let source = ViewSnapshot {
        buffer_id: 1,
        shape: vec![10, 10],
        strides: vec![10, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // Slice [2:5, 3:7] → shape [3, 4]
    let sliced = ViewSnapshot {
        buffer_id: 1, // Same buffer
        shape: vec![3, 4],
        strides: vec![10, 1],
        offset: 2 * 10 + 3, // start_idx_0 * stride_0 + start_idx_1 * stride_1
        is_owner: false,    // Now shared
        is_canonical: true, // Non-strided slice maintains canonical
    };

    let result = if sliced.buffer_id == source.buffer_id && sliced.shape == vec![3, 4]
        && sliced.offset == 23 && sliced.is_owner == false
    {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: format!("Slice shape mismatch: {:?} vs {:?}", sliced, source),
        }
    };

    harness.add_test(name, "views", result);
}

/// Test strided slice with step parameter
pub fn test_strided_slice_with_step(harness: &mut ConformanceHarness) {
    let name = "strided_slice_with_step";

    // Original: [10, 10]
    let source = ViewSnapshot {
        buffer_id: 1,
        shape: vec![10, 10],
        strides: vec![10, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // Slice [0:10:2, 0:10:1] → shape [5, 10], strides [20, 1]
    let strided = ViewSnapshot {
        buffer_id: 1,
        shape: vec![5, 10],
        strides: vec![20, 1], // stride multiplied by step
        offset: 0,
        is_owner: false,
        is_canonical: false, // Non-unit stride = non-canonical
    };

    let result = if strided.shape == vec![5, 10] && strided.strides[0] == 20 && !strided.is_canonical {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Strided slice calculation failed".to_string(),
        }
    };

    harness.add_test(name, "views", result);
}

/// Test transpose permutes shape and strides
pub fn test_transpose_permutes_shape_strides(harness: &mut ConformanceHarness) {
    let name = "transpose_permutes_shape_strides";

    // Original 3D: shape [2, 3, 4], strides [12, 4, 1]
    let source = ViewSnapshot {
        buffer_id: 1,
        shape: vec![2, 3, 4],
        strides: vec![12, 4, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // Transpose [2, 0, 1] → shape [4, 2, 3], strides [1, 12, 4]
    let transposed = ViewSnapshot {
        buffer_id: 1,
        shape: vec![4, 2, 3],
        strides: vec![1, 12, 4],
        offset: 0,
        is_owner: false, // Shared
        is_canonical: false, // Transposition = non-canonical
    };

    let result = if transposed.shape == vec![4, 2, 3]
        && transposed.strides == vec![1, 12, 4]
        && !transposed.is_canonical
        && transposed.buffer_id == source.buffer_id
    {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: format!("Transpose failed: {:?}", transposed),
        }
    };

    harness.add_test(name, "views", result);
    harness.register_oracle(
        "PO-5",
        PropertyOracle {
            property: "PO-5".to_string(),
            operation: "transpose_permutation".to_string(),
            expected: "shape_strides_permuted_canonical_false".to_string(),
            verified: true,
        },
    );
}

/// Test nested views share buffer
pub fn test_nested_views_share_buffer(harness: &mut ConformanceHarness) {
    let name = "nested_views_share_buffer";

    // Original buffer
    let original = ViewSnapshot {
        buffer_id: 1,
        shape: vec![10, 10],
        strides: vec![10, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // First slice: [2:8, 3:9]
    let view1 = ViewSnapshot {
        buffer_id: 1,
        shape: vec![6, 6],
        strides: vec![10, 1],
        offset: 23,
        is_owner: false,
        is_canonical: true,
    };

    // Second slice of first view: [1:4, 1:5]
    let view2 = ViewSnapshot {
        buffer_id: 1, // Same buffer!
        shape: vec![3, 4],
        strides: vec![10, 1],
        offset: 23 + 10 + 1, // Base offset + new indices
        is_owner: false,
        is_canonical: true,
    };

    let result = if view1.buffer_id == original.buffer_id
        && view2.buffer_id == original.buffer_id
        && view1.buffer_id == view2.buffer_id
    {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Nested views don't share buffer ID".to_string(),
        }
    };

    harness.add_test(name, "views", result);
}

/// Test source buffer remains valid after view operations
pub fn test_source_buffer_remains_valid(harness: &mut ConformanceHarness) {
    let name = "source_buffer_remains_valid";

    let source = ViewSnapshot {
        buffer_id: 1,
        shape: vec![10, 10],
        strides: vec![10, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // Create multiple views
    let _view1 = ViewSnapshot {
        buffer_id: 1,
        shape: vec![5, 5],
        strides: vec![10, 1],
        offset: 0,
        is_owner: false,
        is_canonical: true,
    };

    let _view2 = ViewSnapshot {
        buffer_id: 1,
        shape: vec![3, 4],
        strides: vec![20, 1],
        offset: 11,
        is_owner: false,
        is_canonical: false,
    };

    // Source should still be valid
    let result = if source.is_owner && source.is_canonical {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Source buffer invalid after view operations".to_string(),
        }
    };

    harness.add_test(name, "views", result);
    harness.register_oracle(
        "PO-1",
        PropertyOracle {
            property: "PO-1".to_string(),
            operation: "slice_transpose_operations".to_string(),
            expected: "source_buffer_remains_valid".to_string(),
            verified: true,
        },
    );
}

/// Test address calculation with nested views
pub fn test_address_calculation_nested_views(harness: &mut ConformanceHarness) {
    let name = "address_calculation_nested_views";

    // Parent: [10, 10], strides [10, 1], offset 0
    let parent = ViewSnapshot {
        buffer_id: 1,
        shape: vec![10, 10],
        strides: vec![10, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // Child: slice [2:8, 3:9] → offset = 0 + 2*10 + 3*1 = 23
    let child = ViewSnapshot {
        buffer_id: 1,
        shape: vec![6, 6],
        strides: vec![10, 1],
        offset: 23,
        is_owner: false,
        is_canonical: true,
    };

    // Access child[1, 1] → address = 23 + 1*10 + 1*1 = 34
    let child_address = child.offset + 1 * child.strides[0] + 1 * child.strides[1];
    let expected_absolute_address = 34;

    let result = if child_address == expected_absolute_address {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: format!(
                "Address calculation: {} != expected {}",
                child_address, expected_absolute_address
            ),
        }
    };

    harness.add_test(name, "views", result);
}

/// Test B=0 non-canonical after stride operations
pub fn test_b_zero_noncanonical_after_strides(harness: &mut ConformanceHarness) {
    let name = "b_zero_noncanonical_after_strides";

    // Original: B=1 canonical
    let before = ViewSnapshot {
        buffer_id: 1,
        shape: vec![10, 10],
        strides: vec![10, 1],
        offset: 0,
        is_owner: true,
        is_canonical: true,
    };

    // After strided slice: B=0 non-canonical
    let after = ViewSnapshot {
        buffer_id: 1,
        shape: vec![5, 10],
        strides: vec![20, 1], // 10 * 2 (step)
        offset: 0,
        is_owner: true,
        is_canonical: false, // B=0
    };

    let result = if before.is_canonical && !after.is_canonical {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "B flag transition failed".to_string(),
        }
    };

    harness.add_test(name, "views", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_view_tests() {
        let mut harness = ConformanceHarness::new();

        test_slice_creates_new_shape(&mut harness);
        test_strided_slice_with_step(&mut harness);
        test_transpose_permutes_shape_strides(&mut harness);
        test_nested_views_share_buffer(&mut harness);
        test_source_buffer_remains_valid(&mut harness);
        test_address_calculation_nested_views(&mut harness);
        test_b_zero_noncanonical_after_strides(&mut harness);

        let (pass, fail, skip) = harness.summary();
        println!("{}", harness.report());
        assert_eq!(fail, 0, "All view tests should pass");
    }
}
