# Sparse Router

Locality-aware routing and tensor operations, with a Lean model and a standalone
C reference. The supported release candidate surfaces are `rust/tensor-hw/src/`
and `c-reference/`. Licensing reconciliation and target-platform CI are required
before publishing a production release. See [existing licenses](LICENSE-NOTICE.md).

## Build and verify

Prerequisites: Rust 1.85.1 with rustfmt/clippy, a C11 compiler, Python 3, and Lean
4.19.0 through Lake. Validation is CPU-only; LM Studio and a GPU are unnecessary.

```sh
python3 scripts/verify.py
# Or individual checks:
make test
cargo test --locked --manifest-path rust/tensor-hw/Cargo.toml --all-targets --all-features
cd formal && lake build
```

On Windows, pass executable paths through `--cc`, `--cargo` and `--lake`.
For Zig, also pass `--cc-arg cc`. Verification builds sequentially with one Rust
build job. CI additionally runs C address and undefined-behavior sanitizers.

## Behavior

The C router tries a region owner, then up to 16 sparse candidates, then a
least-loaded worker. Full queues or invalid inputs return `WORKER_NONE`; the
caller retains the task for retry. External serialization is required.

The Rust router returns a worker decision with deterministic ties. Configured
region candidates use bounded traversal; absent topology uses a global scan.
It returns an error when no worker is registered. Tensor writes require
`&mut Buffer`, exclusive storage and validated metadata. Materialization copies
strided data transactionally into a distinct buffer.

See [production contracts and migration notes](docs/PRODUCTION.md) for exact
API changes, checks and limits. Source files carry attribution and licensing
notices, checked by `scripts/check_headers.py`.

## Formal and experimental components

Lean proves properties of its decision model, deque and scoring definitions.
It does not certify the executable implementations or prove scheduler liveness.
The legacy FFI/Kani files, Pascal references, transformer code and illustrative
conformance oracles are retained for development and are not certified by the
supported Rust/C test suite. There is no verified Rust-versus-Pascal comparison
or universal throughput guarantee in this checkout.
