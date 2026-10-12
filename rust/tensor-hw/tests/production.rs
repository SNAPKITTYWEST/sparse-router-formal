// SPDX-License-Identifier: GPL-3.0-or-later OR Apache-2.0
// Copyright (C) 2026 SNAPKITTYWEST / Sovereign Kernel Project
// CLONE_GATE: sparse-router-formal::rust::production-tests
use tensor_hw::{
    buffer::Buffer,
    memory::MemoryLayout,
    routing::{RegionId, SparseRouter, Task, WorkerId},
    TensorHardware,
};

fn task() -> Task {
    Task {
        id: 1,
        region: RegionId(7),
        size: 32,
    }
}

#[test]
fn alias_prevents_write_but_allows_read() {
    let mut buffer = Buffer::allocate(1, 4).unwrap();
    buffer.set(0, 7.0).unwrap();
    let alias = buffer.clone();
    assert!(buffer.set(0, 8.0).is_err());
    assert_eq!(alias.get_unchecked(0), 7.0);
    drop(alias);
    buffer.set(0, 9.0).unwrap();
    assert!(buffer.set(4, 1.0).is_err());
}

#[test]
fn empty_router_fails_without_inventing_a_worker() {
    let mut router = SparseRouter::new();
    assert!(router.route(&task()).is_err());
    assert_eq!(router.stats(), (0, 0));
    assert!(router.register_worker(WorkerId(0), f32::NAN).is_err());
    assert!(router.register_worker(WorkerId(0), -1.0).is_err());
}

#[test]
fn sparse_candidates_and_ties_are_deterministic() {
    let mut router = SparseRouter::new();
    for id in (0..20).rev() {
        router.register_worker(WorkerId(id), 1.0).unwrap();
    }
    assert_eq!(router.route(&task()).unwrap(), WorkerId(0));
    router.clear_region(task().region);
    router
        .set_region_candidates(task().region, vec![WorkerId(9), WorkerId(7)])
        .unwrap();
    assert_eq!(router.route(&task()).unwrap(), WorkerId(7));
    assert!(router
        .set_region_candidates(RegionId(8), vec![WorkerId(7); 17])
        .is_err());
    assert!(router
        .set_region_candidates(RegionId(8), vec![WorkerId(100)])
        .is_err());
}

#[test]
fn malformed_metadata_and_overflow_fail_closed() {
    let mut hw = TensorHardware::new();
    assert!(hw.allocate(vec![usize::MAX, 2]).is_err());
    assert!(MemoryLayout::new(8, 0, vec![2, 2], vec![100, 1]).is_err());
    let (buffer, mut view) = hw.allocate(vec![2, 2]).unwrap();
    view.strides.clear();
    assert!(hw.get(&view, &buffer, &[0, 0]).is_err());
    assert!(view.transpose(&[1, 0]).is_err());
}

#[test]
fn materialization_preserves_values_and_uses_distinct_ids() {
    let mut hw = TensorHardware::new();
    let (mut source, view) = hw.allocate(vec![2, 3]).unwrap();
    for i in 0..2 {
        for j in 0..3 {
            hw.set(&view, &mut source, &[i, j], (i * 3 + j) as f32)
                .unwrap();
        }
    }
    let (other, _) = hw.allocate(vec![1]).unwrap();
    let transposed = hw.transpose(&view, &source, &[1, 0]).unwrap();
    let materialized = hw.materialize(&transposed, &source).unwrap();
    assert_ne!(materialized.new_buffer.id, source.id);
    assert_ne!(materialized.new_buffer.id, other.id);
    for i in 0..3 {
        for j in 0..2 {
            assert_eq!(
                hw.get(&materialized.new_view, &materialized.new_buffer, &[i, j])
                    .unwrap(),
                (j * 3 + i) as f32
            );
        }
    }
    let mut malformed = transposed.clone();
    malformed.strides[0] = usize::MAX;
    assert!(hw.materialize(&malformed, &source).is_err());
}

#[test]
fn sliced_stride_metadata_requires_materialization_before_reshape() {
    let mut hw = TensorHardware::new();
    let (mut buffer, view) = hw.allocate(vec![3, 4]).unwrap();
    hw.set(&view, &mut buffer, &[1, 1], 42.0).unwrap();
    let sliced = hw.slice(&view, &buffer, &[(1, 3, 1), (1, 3, 1)]).unwrap();
    assert!(!sliced.is_contiguous());
    let mut forged = sliced.clone();
    forged.ownership.is_canonical = true;
    assert!(hw.reshape(&forged, &buffer, vec![4]).is_err());
    assert!(hw.reshape(&sliced, &buffer, vec![4]).is_err());
    let output = hw.materialize(&sliced, &buffer).unwrap();
    assert_eq!(
        hw.get(&output.new_view, &output.new_buffer, &[0, 0])
            .unwrap(),
        42.0
    );
    assert!(hw.slice(&view, &buffer, &[(1, 1, 1), (0, 4, 1)]).is_err());
}

#[test]
fn transforms_reject_a_descriptor_bound_to_another_buffer() {
    let mut hw = TensorHardware::new();
    let (_, view) = hw.allocate(vec![2, 2]).unwrap();
    let (other, _) = hw.allocate(vec![2, 2]).unwrap();
    assert!(hw.slice(&view, &other, &[(0, 2, 1), (0, 2, 1)]).is_err());
    assert!(hw.transpose(&view, &other, &[1, 0]).is_err());
    assert!(hw.reshape(&view, &other, vec![4]).is_err());
}
