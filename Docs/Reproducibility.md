# Reproducibility and Determinism in HyperStream

This document captures the guarantees and validation workflow for deterministic outputs across operating systems and compilers.

## Guarantees (Phase C)

- Golden HSER1 serialization fixtures: byte-for-byte identical across platforms.
- Encoder determinism: fixed-seed encoders produce identical bit patterns across platforms.
  - ItemMemory (seeded)
  - SymbolEncoder (including role rotations)
  - ThermometerEncoder (deterministic input values)
  - RandomProjectionEncoder (seeded)

## Canonical artifacts committed

- tests/golden/hser1/*.hser1: canonical HSER1 binary fixtures with SHA-256 manifest
- tests/golden/encoder_hashes.json: canonical 64-bit FNV-1a hashes over encoder word buffers

Platform identifiers used in tests:

- windows-msvc
- linux-gcc
- macos-clang

## How tests validate determinism

- Golden serialization tests load committed .hser1 fixtures, deserialize, re-serialize, and assert byte equality. A manifest test verifies SHA-256 for each fixture.
- Encoder determinism tests compute per-encoder 64-bit FNV-1a hashes and assert equality with committed values from tests/golden/encoder_hashes.json for the current platform.
- Backend parity: CI builds and runs the determinism tests twice — once with default backends (SIMD enabled when available) and once with scalar-only by configuring CMake with `-DHYPERSTREAM_FORCE_SCALAR=ON`. Hashes must match in both modes.

## Provenance and verification in CI

- The Golden Compatibility workflow prints compiler versions and key CMake cache entries from both builds (default and force-scalar):
  - `CMAKE_CXX_COMPILER`, `CMAKE_CXX_FLAGS`, `CMAKE_BUILD_TYPE`, `HYPERSTREAM_FORCE_SCALAR`
- Tests additionally record a `backend` property ("simd" or "scalar") for each determinism test to aid debugging.

## Regenerating encoder hashes (when legitimately needed)

1) Build tests (Release recommended):
   - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   - cmake --build build --config Release --parallel 4 --target encoder_determinism_tests

2) Run the disabled helper to dump hashes for the local platform/compiler:
   - build/tests/Release/encoder_determinism_tests --gtest_also_run_disabled_tests \
     --gtest_filter=EncoderDeterminism.DISABLED_DumpEncoderHashes

   You should see lines like:
   {"encoder":"ItemMemory","dim":256,"seed":"0x123456789abcdef0","platform":"windows-msvc","hash":"0x..."}

3) Update tests/golden/encoder_hashes.json entries for your platform as needed (only do this when the change is intentional and code changes justify it). In normal operation, these should remain stable across toolchains.

4) Commit tests-only changes and verify CI Golden Compatibility job is green on ubuntu-latest, windows-latest, macos-14.

## Regenerating HSER1 fixtures

A disabled generator exists in golden_serialization_tests to produce fixtures in tests/golden/hser1/ if ever needed. Use with caution and only under explicit review:

- build/tests/Release/golden_serialization_tests --gtest_also_run_disabled_tests \
  --gtest_filter=GoldenSerialization.DISABLED_GenerateHser1Fixtures

Recompute and update SHA-256 in tests/golden/hser1/manifest.json if fixtures change.

## Notes

- No library code changes are required for determinism tests; the scope is tests and docs only.
- The reference hashes are expected to be identical across platforms; per-platform entries exist to make platform provenance explicit and ease debugging if a divergence is detected by CI.

## Streaming determinism (Phase E)

- Canonical event schema (NDJSON per line): { v:int, seq:uint64, src:string, eid:string, kind:string, ts_ms:int64, payload:object }
  - kind ∈ {"symbol","numeric","vector","label"}
  - payload by kind: {sym:string} | {val:number} | {vec:[number,...]} | {label:string}
- Total-order ingestion invariant: processing is equivalent to ingesting events sorted by (seq asc, src asc, eid asc). ts_ms is informational only.
- Chunking/interleave invariance: For a fixed canonical stream and seed, final state and checkpoints are identical for chunk sizes {1,8,64,randomized} and for pre-merged vs sorted-merge ingestion.
- Epoch/window boundaries: Deterministic checkpoints every K events (K=16 in tests). No implicit flushes elsewhere.
- Idempotency: duplicate (seq,src,eid) is processed at most once (tests ensure duplicate lines cause no net change).
- Vector normalization: vector payloads are accepted at arbitrary length; encoders handle length deterministically (tests use truncation/padding rule in harness where needed).
- Backends/platforms parity: Results match across default (SIMD) and force-scalar builds and across ubuntu-latest, windows-2022, macos-14.
- Golden artifacts:
  - tests/golden/streaming_events.ndjson: canonical stream (fixed seed)
  - tests/golden/streaming_hashes.json: reference 64-bit FNV-1a hashes of checkpoint and final states

### Regenerating streaming hashes
1) Build tests:
   - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   - cmake --build build --config Release --parallel 4 --target streaming_determinism_tests
2) Dump hashes using the disabled helper (includes checkpoints and final):
   - build/tests/Release/streaming_determinism_tests --gtest_also_run_disabled_tests \
     --gtest_filter=StreamingDeterminism.DISABLED_DumpStreamingHashes
3) Update tests/golden/streaming_hashes.json with the emitted values. Keep identical across platforms.


## Perf CI variance-bounds and baseline hygiene (Phase D)

- Variance-bounds enforcement (in validator): stdev/mean thresholds per OS when samples ≥ 3
  - Linux (ubuntu-latest): ≤ 0.10
  - macOS (macos-14): ≤ 0.20
  - Windows (windows-2022): ≤ 0.25
- Aggregates + provenance: the validator emits perf_agg.ndjson with per-group aggregates and provenance fields
  - runner_os, image_os, image_version, cmake_cxx_compiler, cmake_cxx_compiler_version
- Workflow: SAMPLES ≥ 3; NDJSON from benches is aggregated by scripts/bench_check.py, which enforces tolerances and variance-bounds
- Baseline refresh (windows-2022): when windows-latest advances images, keep perf job pinned to windows-2022 until deliberate re-baselining; capture fresh baselines from multiple runs and store medians
