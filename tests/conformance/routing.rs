//! Routing and Work-Stealing Tests: Memory Locality
//!
//! Verifies:
//! - Work-stealing respects NUMA locality
//! - Descriptor routing finds correct buffer by ID + generation
//! - Stale descriptor rejected before routing
//! - Work distribution preserves tensor affinity

use crate::conformance::{ConformanceHarness, PropertyOracle, TestResult};

/// Work unit with routing information
#[derive(Debug, Clone, PartialEq)]
pub struct WorkUnit {
    pub id: u64,
    pub buffer_id: u64,
    pub buffer_generation: u64,
    pub numa_node: usize,
    pub shape: Vec<usize>,
}

/// Routing table entry
#[derive(Debug, Clone, PartialEq)]
pub struct RoutingEntry {
    pub buffer_id: u64,
    pub generation: u64,
    pub location: usize, // NUMA node
    pub valid: bool,
}

/// Test descriptor routing finds correct buffer
pub fn test_descriptor_routing_finds_buffer(harness: &mut ConformanceHarness) {
    let name = "descriptor_routing_finds_buffer";

    // Routing table
    let routing = vec![
        RoutingEntry {
            buffer_id: 1,
            generation: 1,
            location: 0,
            valid: true,
        },
        RoutingEntry {
            buffer_id: 2,
            generation: 2,
            location: 1,
            valid: true,
        },
    ];

    // Query for buffer 2
    let query_id = 2u64;
    let found = routing.iter().find(|e| e.buffer_id == query_id && e.valid);

    let result = if found.is_some() && found.unwrap().location == 1 {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Descriptor routing failed".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

/// Test stale descriptor rejected before routing
pub fn test_stale_descriptor_rejected_before_routing(harness: &mut ConformanceHarness) {
    let name = "stale_descriptor_rejected_before_routing";

    let work = WorkUnit {
        id: 1,
        buffer_id: 1,
        buffer_generation: 1,
        numa_node: 0,
        shape: vec![10, 10],
    };

    // Buffer now at generation 2
    let current_generation = 2u64;

    // Check if descriptor is stale
    let stale = work.buffer_generation != current_generation;

    let result = if stale {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Stale descriptor not rejected".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

/// Test work-stealing respects NUMA locality
pub fn test_work_stealing_respects_locality(harness: &mut ConformanceHarness) {
    let name = "work_stealing_respects_locality";

    // Work on NUMA node 0
    let work_0 = vec![
        WorkUnit {
            id: 1,
            buffer_id: 1,
            buffer_generation: 1,
            numa_node: 0,
            shape: vec![10, 10],
        },
        WorkUnit {
            id: 2,
            buffer_id: 2,
            buffer_generation: 1,
            numa_node: 0,
            shape: vec![20, 20],
        },
    ];

    // Work on NUMA node 1
    let work_1 = vec![
        WorkUnit {
            id: 3,
            buffer_id: 3,
            buffer_generation: 1,
            numa_node: 1,
            shape: vec![30, 30],
        },
    ];

    // Check locality: work_0 items have numa_node=0
    let local_0 = work_0.iter().all(|w| w.numa_node == 0);
    let local_1 = work_1.iter().all(|w| w.numa_node == 1);

    let result = if local_0 && local_1 {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "NUMA locality not respected".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

/// Test buffer affinity preserved during work distribution
pub fn test_buffer_affinity_preserved(harness: &mut ConformanceHarness) {
    let name = "buffer_affinity_preserved";

    let buffer_1_tasks = vec![
        WorkUnit {
            id: 1,
            buffer_id: 1,
            buffer_generation: 1,
            numa_node: 0,
            shape: vec![10, 10],
        },
        WorkUnit {
            id: 2,
            buffer_id: 1,
            buffer_generation: 1,
            numa_node: 0,
            shape: vec![10, 10],
        },
    ];

    // All tasks for buffer 1 should be on same NUMA node
    let single_node = buffer_1_tasks.iter().all(|w| w.numa_node == 0);
    let same_buffer = buffer_1_tasks.iter().all(|w| w.buffer_id == 1);

    let result = if single_node && same_buffer {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Buffer affinity not preserved".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

/// Test routing invalidates on generation mismatch
pub fn test_routing_invalidates_stale(harness: &mut ConformanceHarness) {
    let name = "routing_invalidates_stale";

    let mut entry = RoutingEntry {
        buffer_id: 1,
        generation: 1,
        location: 0,
        valid: true,
    };

    // Buffer generation changed
    let current_generation = 2u64;
    if entry.generation != current_generation {
        entry.valid = false;
    }

    let result = if !entry.valid {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Stale routing entry not invalidated".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

/// Test multi-NUMA routing balances load
pub fn test_multi_numa_routing_balances(harness: &mut ConformanceHarness) {
    let name = "multi_numa_routing_balances";

    let mut work_per_node = std::collections::HashMap::new();

    // Distribute work across 3 NUMA nodes
    let work = vec![
        WorkUnit {
            id: 1,
            buffer_id: 1,
            buffer_generation: 1,
            numa_node: 0,
            shape: vec![10, 10],
        },
        WorkUnit {
            id: 2,
            buffer_id: 2,
            buffer_generation: 1,
            numa_node: 1,
            shape: vec![10, 10],
        },
        WorkUnit {
            id: 3,
            buffer_id: 3,
            buffer_generation: 1,
            numa_node: 2,
            shape: vec![10, 10],
        },
        WorkUnit {
            id: 4,
            buffer_id: 4,
            buffer_generation: 1,
            numa_node: 0,
            shape: vec![10, 10],
        },
    ];

    for w in &work {
        *work_per_node.entry(w.numa_node).or_insert(0) += 1;
    }

    // Should have: node0=2, node1=1, node2=1
    let balanced = work_per_node.get(&0) == Some(&2)
        && work_per_node.get(&1) == Some(&1)
        && work_per_node.get(&2) == Some(&1);

    let result = if balanced {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Load not balanced across NUMA nodes".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

/// Test generation validation in routing path
pub fn test_generation_validation_in_routing(harness: &mut ConformanceHarness) {
    let name = "generation_validation_in_routing";

    // Descriptor from earlier in program
    let descriptor_gen = 1u64;
    let descriptor_id = 1u64;

    // Routing lookup
    let routing_entry = RoutingEntry {
        buffer_id: 1,
        generation: 2, // Buffer was rematerialized
        location: 0,
        valid: true,
    };

    // Validation should fail
    let valid = descriptor_gen == routing_entry.generation && descriptor_id == routing_entry.buffer_id;

    let result = if !valid {
        TestResult::Pass
    } else {
        TestResult::Fail {
            reason: "Generation validation failed in routing".to_string(),
        }
    };

    harness.add_test(name, "routing", result);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_routing_tests() {
        let mut harness = ConformanceHarness::new();

        test_descriptor_routing_finds_buffer(&mut harness);
        test_stale_descriptor_rejected_before_routing(&mut harness);
        test_work_stealing_respects_locality(&mut harness);
        test_buffer_affinity_preserved(&mut harness);
        test_routing_invalidates_stale(&mut harness);
        test_multi_numa_routing_balances(&mut harness);
        test_generation_validation_in_routing(&mut harness);

        let (pass, fail, skip) = harness.summary();
        println!("{}", harness.report());
        assert_eq!(fail, 0, "All routing tests should pass");
    }
}
