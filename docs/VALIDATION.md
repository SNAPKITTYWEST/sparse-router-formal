# Local validation, 12 October 2026

Upstream checkout: `5e292ee830fd54e868ea74c501a7ccd44e6318d2`.
Local branch: `production-hardening`. Changes have not been published.

The sequential CPU-only `scripts/verify.py` run completed on Windows with Rust
1.85.1, Lean 4.19.0 and Zig 0.14.0 as the C compiler. Builds used one Rust job
and below-normal process priority. LM Studio inference was stopped at the owner's
request and no GPU model was used for these checks.

| Check | Result |
| --- | --- |
| Source attribution/licensing notices | 67 files checked, no missing notices |
| C regression tests | 10 passed |
| Rust debug tests | 49 unit + 7 regression tests passed |
| Rust release tests | 49 unit + 7 regression tests passed |
| Rust format and Clippy | Passed, warnings treated as errors |
| Rust example | Executed successfully |
| Lean model | All imported modules compiled |
| Proof-source check | No empty modules, admissions or added axioms found |
| Git whitespace check | Passed |

The inherited Rust documentation snippet remains ignored as a test; the
executable example provides the tested API usage. Linux sanitizer CI is configured
but has not run on GitHub in this checkout. Pascal, legacy FFI, Kani and the
illustrative conformance oracles were not executed by this validation.

This evidence establishes the checks above, not whole-repository production
certification. Outstanding release gates are documented in `PRODUCTION.md` and
`LICENSE-NOTICE.md`. Package publishing remains disabled until those gates clear.
