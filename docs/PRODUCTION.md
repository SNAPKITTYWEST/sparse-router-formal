# Production verification and supported surface

This checkout hardens the Rust library and standalone C reference. It is a
release candidate pending licensing reconciliation and CI on the target platform.

## Rust API changes

`TensorHardware::set` now takes `&mut Buffer`. `Buffer::set` checks bounds and
uses `Arc::get_mut`; cloned/shared storage cannot be written. Buffer storage is
private. Materialization validates all source metadata and copies every element
or returns an error. Allocations and materializations use one checked ID allocator.
Malformed shapes, stride ranks, address overflow and invalid locality metrics are
rejected. `SparseRouter::route` and `register_worker` now return `Result`.
An empty router returns an error without recording a successful route.

`set_region_candidates` admits at most 16 distinct registered workers. With
configured candidates, scoring traverses that list; without a topology, scoring
scans all registered workers. Ties choose the smaller worker ID. The Rust router
selects workers; it does not enqueue tasks or implement a concurrent executor.
Callers maintain worker queue metrics. Clearing a region releases its owner record.

Tensor views hold descriptors, not owning `Arc` references: callers retain the
associated `Buffer`. Rust references and checked buffer identity govern access.
This is not an external device-memory registry or a hardware allocator.

## C contract

`raw_route` returns `WORKER_NONE` on malformed inputs or unavailable capacity.
The caller retains that task for retry. Ownership and access counters commit only
after successful enqueue. A full region owner permits another available worker.
The caller must initialize the state and externally serialize routing, stealing
and deque operations. This reference is not a lock-free concurrent deque.

## Verification

Run `python3 scripts/check_headers.py`, `make test`, Rust debug/release tests,
`cargo fmt --check`, `cargo clippy -- -D warnings`, and `cd formal && lake build`.
CI additionally runs the C suite with address and undefined-behavior sanitizers.
Lean is pinned by `formal/lean-toolchain`; Rust dependencies are lockfile-pinned.
GPU inference is not required by any check.

The Lean model proves decision uniqueness, bounded candidate lists, deque
properties, scoring lemmas, and a characterization of zero unowned regions.
It does not prove that executable Rust or C refines that model, scheduler liveness,
queue admission under exhaustion, or convergence within a fixed number of calls.

Legacy files in `rust/tensor-hw/ffi/`, the Kani harnesses, and Pascal transformer
sources are outside the supported Rust crate. The old `tests/conformance/`
contains illustrative oracle logic, not an executed Rust-versus-Pascal comparison.
They require integration and independent verification before release claims.

## Development workflow

Codex in DataGrip orchestrated inspection, implementation and verification.
An initial call through the existing LM Studio supervisor timed out without a
usable review. At the owner's request LM Studio models were unloaded and its
server stopped; remaining verification uses CPU tools with one Rust build job
and below-normal priority. No successful local-model review is claimed.
