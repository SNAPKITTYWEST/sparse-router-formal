//! Core tensor hardware operations example
//!
//! Demonstrates:
//! - Allocation of exclusive canonical tensors
//! - Element access with validation
//! - Shared views (zero-copy)
//! - Slicing with non-destructive ranges
//! - Transpose (creates non-canonical strides)
//! - Materialization (transactional copy-on-write)

use tensor_hw::*;

fn main() -> Result<()> {
    println!("NraayTensor Hardware Layer Examples");
    println!("===================================\n");

    let mut hw = TensorHardware::new();

    // Example 1: Allocate and set values
    println!("1. Allocate 10x10 tensor");
    let (buffer, view) = hw.allocate(vec![10, 10])?;
    println!("   Shape: {:?}", view.shape);
    println!("   Element count: {}", view.element_count());
    println!("   Exclusive: {}, Contiguous: {}\n", view.is_exclusive(), view.is_contiguous());

    // Example 2: Element access
    println!("2. Set and get elements");
    hw.set(&view, &buffer, &[2, 3], 42.0)?;
    let val = hw.get(&view, &buffer, &[2, 3])?;
    println!("   Set [2,3] = 42.0");
    println!("   Get [2,3] = {}\n", val);

    // Example 3: Shared view (zero-copy)
    println!("3. Create shared view (zero-copy Arc::clone)");
    let shared = hw.create_view(&view, &buffer)?;
    println!("   Shared exclusive: {}", shared.is_exclusive());
    println!("   Can read from shared: OK");
    let result = hw.set(&shared, &buffer, &[2, 3], 50.0);
    println!("   Try to write to shared: {:?}\n", result);

    // Example 4: Slice operation
    println!("4. Slice [2:5, 3:7]");
    let sliced = hw.slice(&view, &buffer, &[(2, 5, 1), (3, 7, 1)])?;
    println!("   New shape: {:?}", sliced.shape);
    println!("   Exclusive: {}", sliced.is_exclusive());
    println!("   Contiguous: {}\n", sliced.is_contiguous());

    // Example 5: Transpose (non-canonical)
    println!("5. Transpose dimensions [1, 0]");
    let transposed = hw.transpose(&view, &buffer, &[1, 0])?;
    println!("   New shape: {:?}", transposed.shape);
    println!("   Contiguous (canonical): {}\n", transposed.is_contiguous());

    // Example 6: Materialize non-canonical
    println!("6. Materialize transposed tensor");
    let mat_result = hw.materialize(&transposed, &buffer)?;
    println!("   New buffer ID: {}", mat_result.new_buffer.id);
    println!("   New generation: {}", mat_result.new_buffer.generation);
    println!("   New exclusive: {}", mat_result.new_view.is_exclusive());
    println!("   New contiguous: {}", mat_result.new_view.is_contiguous());
    println!("   Source buffer still valid: {}\n", buffer.is_exclusive());

    // Example 7: Address calculation
    println!("7. Address calculation with strides");
    let addr = view.address_for(&[3, 5])?;
    println!("   Address for [3, 5] in 10x10 (canonical): {}", addr);
    println!("   (Offset 0 + 3*10 + 5*1 = 35)\n");

    // Example 8: Sparse routing
    println!("8. Work-stealing router (RAW_ROUTE)");
    let mut router = SparseRouter::new();
    router.register_worker(WorkerId(0), 0.5);
    router.register_worker(WorkerId(1), 1.5);

    let task1 = Task {
        id: 1,
        region: RegionId(100),
        size: 1024,
    };
    let task2 = Task {
        id: 2,
        region: RegionId(100),
        size: 1024,
    };

    let w1 = router.route(&task1);
    let w2 = router.route(&task2);
    println!("   Task 1 routed to: {:?}", w1);
    println!("   Task 2 (same region) routed to: {:?}", w2);
    println!("   Region consistency: {}\n", w1 == w2);

    // Example 9: Governance validation
    println!("9. Governance validation");
    let mut validator = GovernanceValidator::new();
    let result = validator.validate_view(&view, &buffer);
    println!("   Validate view: {:?}", result.is_ok());
    let result = validator.validate_access(&view, &buffer, &[0, 0], false);
    println!("   Validate read access: {:?}", result.is_ok());
    let result = validator.validate_access(&view, &buffer, &[0, 0], true);
    println!("   Validate write access: {:?}", result.is_ok());
    let result = validator.validate_access(&shared, &buffer, &[0, 0], true);
    println!("   Validate write on shared: {:?}", result.is_err());
    println!("   Operations validated: {}\n", validator.sequence());

    // Example 10: Error handling (fail-closed)
    println!("10. Fail-closed error handling");
    let oob_result = hw.get(&view, &buffer, &[100, 100]);
    println!("   Access [100, 100] (out of bounds): {:?}", oob_result);

    let bad_slice = hw.slice(&view, &buffer, &[(0, 100, 1), (0, 10, 1)]);
    println!("   Slice [0:100, 0:10] (out of bounds): {}", bad_slice.is_err());

    println!("\nAll examples completed successfully!");
    Ok(())
}
