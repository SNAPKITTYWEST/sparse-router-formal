//! Cross-Language Conformance Test Suite
//!
//! Comprehensive test coverage for Rust and Pascal tensor implementations.
//! Verifies binary protocol compatibility, semantic equivalence, and temporal safety.
//!
//! Test categories (~/2000 LOC total):
//! - conformance/buffer_lifetime.rs (T1-T5: Parent-child dropping)
//! - conformance/ownership.rs (Strong/weak refs, B_own transitions)
//! - conformance/views.rs (Slicing, transposing, nested views)
//! - conformance/materialization.rs (B=1 canonical + new buffer)
//! - conformance/generation.rs (Stale descriptor detection)
//! - conformance/governance.rs (State transitions, fail-closed)
//! - routing.rs (Work-stealing with memory locality)
//!
//! Oracle tables for PO-1 through PO-8 (temporal safety properties).

pub mod buffer_lifetime;
pub mod generation;
pub mod governance;
pub mod materialization;
pub mod ownership;
pub mod routing;
pub mod views;

use std::collections::HashMap;
use std::fmt;

/// Test result tracking for conformance testing
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TestResult {
    Pass,
    Fail { reason: String },
    Skip { reason: String },
}

impl fmt::Display for TestResult {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TestResult::Pass => write!(f, "PASS"),
            TestResult::Fail { reason } => write!(f, "FAIL: {}", reason),
            TestResult::Skip { reason } => write!(f, "SKIP: {}", reason),
        }
    }
}

/// Single test case
#[derive(Debug, Clone)]
pub struct TestCase {
    pub name: String,
    pub category: String,
    pub result: TestResult,
}

/// Conformance test harness
pub struct ConformanceHarness {
    tests: Vec<TestCase>,
    oracle_tables: HashMap<String, Vec<PropertyOracle>>,
}

/// Property oracle for temporal safety verification (PO-1 through PO-8)
#[derive(Debug, Clone)]
pub struct PropertyOracle {
    pub property: String,  // PO-1 through PO-8
    pub operation: String, // Specific operation (e.g., parent_allocation_child_view_created)
    pub expected: String,  // Expected behavior
    pub verified: bool,    // Whether oracle was verified
}

impl ConformanceHarness {
    pub fn new() -> Self {
        ConformanceHarness {
            tests: Vec::new(),
            oracle_tables: HashMap::new(),
        }
    }

    /// Add a test case
    pub fn add_test(&mut self, name: &str, category: &str, result: TestResult) {
        self.tests.push(TestCase {
            name: name.to_string(),
            category: category.to_string(),
            result,
        });
    }

    /// Register a temporal safety property oracle
    pub fn register_oracle(&mut self, property: &str, oracle: PropertyOracle) {
        self.oracle_tables
            .entry(property.to_string())
            .or_insert_with(Vec::new)
            .push(oracle);
    }

    /// Generate conformance report
    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("=== CROSS-LANGUAGE CONFORMANCE TEST REPORT ===\n\n");

        let categories: std::collections::BTreeMap<_, _> =
            self.tests.iter().fold(std::collections::BTreeMap::new(), |mut acc, t| {
                acc.entry(&t.category).or_insert_with(Vec::new).push(t);
                acc
            });

        let (total_pass, total_fail, total_skip) = self.summary();
        report.push_str(&format!(
            "SUMMARY: {} pass, {} fail, {} skip ({} total)\n\n",
            total_pass,
            total_fail,
            total_skip,
            self.tests.len()
        ));

        for (category, tests) in categories {
            let pass_count = tests.iter().filter(|t| t.result == TestResult::Pass).count();
            let fail_count = tests.iter().filter(|t| matches!(t.result, TestResult::Fail { .. })).count();
            let skip_count = tests.iter().filter(|t| matches!(t.result, TestResult::Skip { .. })).count();

            report.push_str(&format!(
                "\n## [{}] {}: {}/{} pass\n",
                category,
                category,
                pass_count,
                tests.len()
            ));

            for test in tests {
                let status = match &test.result {
                    TestResult::Pass => "✓",
                    TestResult::Fail { .. } => "✗",
                    TestResult::Skip { .. } => "⊘",
                };
                report.push_str(&format!("  {} {}\n", status, test.name));
                if !matches!(test.result, TestResult::Pass) {
                    report.push_str(&format!("    {}\n", test.result));
                }
            }
        }

        report.push_str("\n=== TEMPORAL SAFETY ORACLES (PO-1 through PO-8) ===\n");
        let mut oracle_keys: Vec<_> = self.oracle_tables.keys().collect();
        oracle_keys.sort();

        for prop in oracle_keys {
            if let Some(oracles) = self.oracle_tables.get(prop) {
                let verified = oracles.iter().filter(|o| o.verified).count();
                report.push_str(&format!(
                    "\n{}: {}/{} verified\n",
                    prop,
                    verified,
                    oracles.len()
                ));

                for oracle in oracles {
                    let status = if oracle.verified { "✓" } else { "✗" };
                    report.push_str(&format!(
                        "  {} {}\n      → {}\n",
                        status, oracle.operation, oracle.expected
                    ));
                }
            }
        }

        report
    }

    /// Get summary statistics
    pub fn summary(&self) -> (usize, usize, usize) {
        let pass = self.tests.iter().filter(|t| t.result == TestResult::Pass).count();
        let fail = self.tests.iter().filter(|t| matches!(t.result, TestResult::Fail { .. })).count();
        let skip = self.tests.iter().filter(|t| matches!(t.result, TestResult::Skip { .. })).count();
        (pass, fail, skip)
    }

    /// Get oracle verification rate
    pub fn oracle_verification_rate(&self) -> (usize, usize) {
        let total_oracles: usize = self.oracle_tables.values().map(|v| v.len()).sum();
        let verified_oracles: usize = self
            .oracle_tables
            .values()
            .flat_map(|v| v.iter().filter(|o| o.verified))
            .count();
        (verified_oracles, total_oracles)
    }
}

impl Default for ConformanceHarness {
    fn default() -> Self {
        Self::new()
    }
}

/// Temporal Safety Properties (PO-1 through PO-8)
///
/// PO-1: Parent buffer allocation creates valid buffer handle
///       Parent remains valid after child view creation
///
/// PO-2: Child drop with active parent
///       Parent dropping with active children: children remain valid, buffer not freed
///
/// PO-3: All children drop, then parent drop
///       When all children and parent drop (refcount → 0), buffer is deallocated
///
/// PO-4: Shared view creation (Arc::clone)
///       B_own: 1 → 0 (exclusive → shared), version++, strong_count++
///
/// PO-5: Transposition or non-canonical strides
///       B_layout: 1 → 0 (canonical → non-canonical), version++
///
/// PO-6: Materialization creates new buffer
///       New buffer_id, generation++, B_own → 1, B_layout → 1
///       Source view remains valid with old buffer_id
///
/// PO-7: Governance fail-closed operations
///       Rank mismatch, out-of-bounds, shared mutation, reshape non-canonical rejected
///
/// PO-8: Generation counter prevents use-after-free
///       Stale descriptor (generation mismatch) rejected
///       ABA problem prevented by generation counter

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_harness_creation() {
        let harness = ConformanceHarness::new();
        let (pass, fail, skip) = harness.summary();
        assert_eq!((pass, fail, skip), (0, 0, 0));
    }

    #[test]
    fn test_summary_counts() {
        let mut harness = ConformanceHarness::new();
        harness.add_test("test1", "category1", TestResult::Pass);
        harness.add_test("test2", "category1", TestResult::Fail {
            reason: "boom".to_string(),
        });
        harness.add_test("test3", "category2", TestResult::Skip {
            reason: "not ready".to_string(),
        });

        let (pass, fail, skip) = harness.summary();
        assert_eq!((pass, fail, skip), (1, 1, 1));
    }

    #[test]
    fn test_oracle_registration() {
        let mut harness = ConformanceHarness::new();
        let oracle = PropertyOracle {
            property: "PO-1".to_string(),
            operation: "test_op".to_string(),
            expected: "test_expected".to_string(),
            verified: true,
        };
        harness.register_oracle("PO-1", oracle.clone());
        harness.register_oracle("PO-1", oracle);

        let (verified, total) = harness.oracle_verification_rate();
        assert_eq!((verified, total), (2, 2));
    }

    #[test]
    fn test_report_generation() {
        let mut harness = ConformanceHarness::new();
        harness.add_test("test1", "conformance", TestResult::Pass);
        harness.add_test("test2", "ownership", TestResult::Fail {
            reason: "mismatch".to_string(),
        });

        let report = harness.report();
        assert!(report.contains("SUMMARY"));
        assert!(report.contains("conformance"));
        assert!(report.contains("ownership"));
    }
}
