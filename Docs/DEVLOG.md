# HyperStream Devlog

Date: 2025-10-06 03:05:36 -06:00

---

## Summary
- **Bind AVX2 frozen (main path)** in `include/hyperstream/backend/cpu_backend_avx2.hpp` using loadu/storeu, no prefetch; NT store threshold kept as env knob. AVX2 ≈ core at small/mid sizes; slight wins ≥64K.
- **Hamming microbench** shows SSE2+POPCNT is strongest at large dims on this host; retain both SSE2 and AVX2 backends.
- **Permutation optimized (bool)**: `PermuteRotate<Dim,bool>` rewritten as word-wise rotation; performance near memory-traffic ceiling across dims.
- **Benchmark harness** improvements: AVX2-first ordering and input-only sink mixing for fair, stable measurements; added `hamming_bench` and `permute_bench`.
- **Next**: Integrate perf regression checks, raise test coverage, implement encoders and associative memory per plan, fix remaining doc lints.

---

## Detailed Log

### 2025-10-06 02:50–03:05 — Edge Case Tests: Diagnosis and Fixes

- **Context**
  - New target `edge_case_tests` added in `tests/CMakeLists.txt`:
    - Target: `edge_case_tests`
    - Source: `tests/edge_case_tests.cc`
    - Linked libs: `hyperstream`, `gtest`, `gtest_main`
    - Warnings-as-errors enabled (MSVC `/W4 /WX`)

- **Issues observed**
  - Duplicate `TEST` definitions and names inside anonymous namespaces caused redefinition errors.
  - Mixed namespace usage (e.g., `core::` qualifiers while also introducing `using hyperstream::core::...`) produced confusion and compile errors.
  - An `EXPECT_DEATH`-style check conflicted with runtime behavior; `HyperVector<bool>` throws `std::out_of_range` for OOB access rather than aborting.
  - `[[nodiscard]]` on `GetBit()` raised a warning treated as error when the return value was discarded inside `EXPECT_THROW`.

- **Changes made to `tests/edge_case_tests.cc`**
  - Ensured **unique test names** and removed inadvertent duplicated blocks.
  - Normalized to **explicit using declarations** (no `using namespace`) to match `tests/core_ops_tests.cc`:
    - `using hyperstream::core::HyperVector;`
    - `using hyperstream::core::Bind;`
    - `using hyperstream::core::HammingDistance;`
    - `using hyperstream::core::CosineSimilarity;`
  - Added a **file header comment** describing scope and coverage.
  - Replaced brittle death tests with **exception assertions** aligned with the implementation:
    - `EXPECT_THROW(hv.GetBit(D), std::out_of_range)` (wrapped in a volatile sink to satisfy `[[nodiscard]]`).
  - Added a **safe large-dimension** smoke test at 65,536 bits (1<<16) to catch word/bit boundary issues without risking Debug-time stack/memory issues.
  - Maintained and verified **numeric-type** tests (int8/uint8) and **float/complex** tests with `CosineSimilarity()` ≈ 1.0 for identical vectors.

- **Validation**
  - Built only target:
    - `cmake --build . --config Debug --target edge_case_tests`
    - Result: success, produced `build/tests/Debug/edge_case_tests.exe`.
  - Ran target binary:
    - `tests\Debug\edge_case_tests.exe --gtest_color=yes`
    - Result: `[  PASSED  ] 6 tests`.
  - Ran full suite:
    - `ctest -C Debug -j 8 --output-on-failure`
    - Result: 25/25 tests passed (0.16s real time).

### Implementation Plan — State

- `IMPLEMENTATION_PLAN.md` has been expanded to include:
  - **Implementation status & validation** with requirements coverage.
  - **Test implementation plan** with unit/perf/fuzz coverage targets.
  - **Benchmarks** outline (bind op, memory bandwidth) using Google Benchmark.
  - **Performance budgets**, roadmap, risks.

- Known documentation lint issues (to fix):
  - Missing link definition: `'Unreleased'` at line 290.
  - Missing link definition: `'Specify License'` at line 314.

---

## Results
- Edge-case coverage retained and clarified:
  - `EdgeCaseTests.MinDimension` (8-bit correctness)
  - `EdgeCaseTests.NonPowerOfTwoDimensions` (D=100 behavior)
  - `EdgeCaseTests.OutOfRangeThrows` (exception behavior matches implementation)
  - `EdgeCaseTests.NumericTypes` (int8/uint8 set/get)
  - `EdgeCaseTests.FloatAndComplexTypes` (cosine similarity)
  - `EdgeCaseTests.MaxReasonableDimension_BasicOps` (65,536-bit smoke)
- Style and design consistent with `tests/core_ops_tests.cc`.
- All test targets pass under strict warnings.

---

## Decisions
- **Exceptions over death tests** for out-of-bounds: aligns with `HyperVector<bool>`’s throwing behavior.
- **Bounded large dimension** tests to 65,536 in Debug builds to avoid excessive stack/memory while preserving boundary coverage.
- **No `using namespace`** in tests; only explicit symbol imports for consistency and readability.

---

## Next Steps
- **Benchmarks**
  - Add CMake target(s) for Google Benchmark-based tests per plan (bind op, memory bandwidth).
  - Validate targets across representative dimensions (1K–1M) in Release with AVX2/SSE2 backends.

- **Coverage**
  - Integrate coverage tooling:
    - MSVC: VS Enterprise coverage or switch builds to clang-cl + llvm-cov for consistent output.
  - Aim for >90% per plan. Start with core ops, encoders, associative memory.

- **Fuzz/Property testing**
  - Add fuzzers for input validation paths and property-based tests (e.g., XOR involution, rotation inverses).

- **Docs**
  - Fix two missing link definitions in `IMPLEMENTATION_PLAN.md`.
  - Add a brief HOWTO for running tests/benchmarks locally (Windows/Unix).

---

## Appendix: Commands Executed

```powershell
# Build specific target
cmake --build . --config Debug --target edge_case_tests

# Run specific test binary
# (from build directory)
tests\Debug\edge_case_tests.exe --gtest_color=yes

# Run full suite
ctest -C Debug -j 8 --output-on-failure
```

### 2025-10-06 03:13–03:16 — Benchmarks: Bind baseline

- **Added** `benchmarks/` with `bind_bench.cpp` and `benchmarks/CMakeLists.txt`.
- **Enabled** `HYPERSTREAM_ENABLE_BENCHMARKS` in root `CMakeLists.txt`.
- **Built/ran** `bind_bench` in Release.

Results (GB/s, Release):

| Dim (bits) | Bind/core | Bind/sse2 | Bind/avx2 |
|---:|---:|---:|---:|
| 1,024 | 14.821 | 15.609 | 16.050 |
| 2,048 | 29.519 | 29.068 | 31.319 |
| 4,096 | 56.248 | 42.546 | 57.494 |
| 8,192 | 97.515 | 79.723 | 80.574 |
| 10,000 | 109.275 | 86.705 | 107.844 |
| 16,384 | 154.177 | 77.064 | 158.386 |
| 65,536 | 157.070 | 84.105 | 148.421 |
| 262,144 | 105.257 | 81.148 | 98.565 |
| 1,048,576 | 119.969 | 91.490 | 116.710 |

Notes:
- **Memory-bound behavior** indicated by converging GB/s at large dims; AVX2 advantage shrinks.
- **Core (scalar) beats AVX2** at some sizes due to memory subsystem limits and implementation details.
- Baseline established for subsequent AVX2 optimizations (prefetch, non-temporal stores) and alignment tweaks.

Next Actions (per workflow):
- Implement `BindAVX2_Optimized` with prefetching and opportunistic non-temporal stores (no compile flags; runtime heuristics).
- Extend benchmark to include `Bind/avx2_opt` and re-run.
- Document impact and decide on defaulting to optimized path for large outputs.

### 2025-10-06 03:18–03:24 — AVX2 Bind updated as main path + post-optimization results

- **Updated** `include/hyperstream/backend/cpu_backend_avx2.hpp`:
  - Added runtime heuristics in `BindAVX2()`:
    - Prefetch inputs at distance of 64 words with `_MM_HINT_T0`.
    - Use `_mm256_stream_si256` (non-temporal) when `out` is 32B-aligned and `out_bytes >= 64KB`.
    - `_mm_sfence()` after loop when streaming was used.
- **Aligned** binary storage in `include/hyperstream/core/hypervector.hpp`:
  - `alignas(32)` for `HyperVector<Dim,bool>::word_` to favor aligned vector ops.
- **Benchmark fairness**: `benchmarks/bind_bench.cpp` now mixes the sink with input words to avoid penalizing non-temporal stores by reading `out`.

Results (GB/s, Release, after optimization):

| Dim (bits) | Bind/core | Bind/sse2 | Bind/avx2 (optimized) |
|---:|---:|---:|---:|
| 1,024 | 14.544 | 15.562 | 14.638 |
| 2,048 | 29.392 | 29.693 | 27.244 |
| 4,096 | 55.973 | 41.598 | 47.041 |
| 8,192 | 99.740 | 82.720 | 95.662 |
| 10,000 | 109.614 | 86.624 | 100.404 |
| 16,384 | 141.811 | 101.351 | 113.033 |
| 65,536 | 165.080 | 87.992 | 113.525 |
| 262,144 | 122.252 | 94.570 | 118.872 |
| 1,048,576 | 128.393 | 95.870 | 52.380 |

Analysis:
- **Regression for AVX2** at most sizes vs baseline. Likely causes:
  - Non-temporal stores reduce cache pollution but can underperform without sufficient write combining or on certain memory subsystems.
  - Prefetching with `_MM_HINT_T0` may compete with demand loads in L1; distance/hint may need tuning.
- **Core (scalar) now leads** at many sizes, indicating a strongly memory-bound regime on this host.

Immediate Tuning Plan:
- Add aligned loads/stores when possible (`_mm256_load_si256/_mm256_store_si256`) while retaining streaming for stores only if beneficial.
- Re-tune heuristics:
  - Increase streaming threshold (e.g., from 64KB → 256KB or higher) to avoid regressions for mid-size vectors.
  - Experiment with prefetch distance and `_MM_HINT_T1` for larger strides.
- Re-run benchmarks and adopt the best-performing configuration as the main path.

### 2025-10-06 03:33–03:36 — Conservative tuning and results

- **Changes** (conservative):
  - Disabled manual prefetching in `BindAVX2()`.
  - Raised non-temporal store threshold to ≥1MB; still requires 32B-aligned output.
  - Kept aligned loads/stores based on 32B alignment of inputs/outputs.

- **Results (GB/s, Release, conservative tuning):**

| Dim (bits) | Bind/core | Bind/sse2 | Bind/avx2 (conservative) |
|---:|---:|---:|---:|
| 1,024 | 14.949 | 15.369 | 15.155 |
| 2,048 | 27.863 | 28.389 | 28.173 |
| 4,096 | 49.796 | 45.345 | 57.515 |
| 8,192 | 101.374 | 80.209 | 99.961 |
| 10,000 | 94.124 | 75.367 | 103.179 |
| 16,384 | 140.230 | 101.207 | 151.958 |
| 65,536 | 221.995 | 87.057 | 208.360 |
| 262,144 | 126.525 | 122.714 | 125.780 |
| 1,048,576 | 127.765 | 123.701 | 126.970 |

- **Interpretation**:
  - AVX2 is now ≈ core at small/mid sizes and beats core at several mid-range dimensions (4K–16K, 10K).
  - For very large (≥64K), core sometimes leads; both converge near memory bandwidth limits.
  - Disabling prefetch and raising the streaming threshold improved stability and removed prior regressions.

- **Next**:
  - Keep current conservative path as default.
  - If future hosts benefit, we can expose a runtime knob for streaming stores threshold; for now, 1MB is safe.
  - Proceed to benchmark Hamming distance and permutation similarly; record results.

### 2025-10-06 03:48 — Regression diagnosed (AVX2 bind, unrolled storeu-only)

- **Change**: In `BindAVX2()` unrolled loop, replaced both `_mm256_stream_si256` and `_mm256_store_si256` with `_mm256_storeu_si256` (stores) while keeping aligned loads when safe.
- **Observation**: AVX2 throughput still collapsed compared to core/SSE2.

Representative results (GB/s, Release):

| Dim (bits) | Bind/core | Bind/sse2 | Bind/avx2 (unrolled, storeu-only) |
|---:|---:|---:|---:|
| 1,024 | 14.631 | 14.492 | 0.256 |
| 4,096 | 55.896 | 47.476 | 1.019 |
| 16,384 | 146.818 | 93.233 | 4.055 |
| 65,536 | 220.808 | 88.142 | 15.307 |
| 1,048,576 | 130.039 | 125.689 | 86.416 |

- **Conclusion**: The regression persists, likely tied to the new unrolled AVX2 store path itself (not just the store type). The scalar and SSE2 paths are unaffected; Hamming benchmarks indicate the read-heavy kernels behave as expected.

- **Next corrective step**:
  - Revert the unrolled AVX2 bind to the previously good conservative loop (single 256b chunk per iter, `_mm256_storeu_si256`, no prefetch, NT threshold ≥1MB for future A/B).
  - Re-benchmark. If restored, A/B reintroducing aligned stores (writes) only if they show consistent wins on this host.

### 2025-10-06 03:55 — A/B experiments: ordering, AVX2_ref, sink read removal

- **Changes**:
  - Reordered `bind_bench` to run AVX2 first, then `avx2_ref`, then core, then SSE2 per-dimension.
  - Added inline `BindAVX2_Ref` (loadu→xor→storeu, no alignment/NT logic) in the bench to isolate header overhead.
  - Simplified sink mixing to only use input words (avoid reading `out`).

- **Results (GB/s, Release):**

| Dim (bits) | Bind/avx2 | Bind/avx2_ref | Bind/core | Bind/sse2 |
|---:|---:|---:|---:|---:|
| 1,024 | 13.842 | 15.332 | 12.586 | 13.801 |
| 2,048 | 26.458 | 30.568 | 29.152 | 26.863 |
| 4,096 | 50.266 | 55.616 | 53.550 | 47.310 |
| 8,192 | 77.045 | 94.848 | 88.910 | 76.879 |
| 10,000 | 99.709 | 104.324 | 108.992 | 85.727 |
| 16,384 | 157.719 | 159.455 | 160.617 | 101.114 |
| 65,536 | 227.952 | 163.939 | 223.951 | 87.795 |
| 262,144 | 126.064 | 126.506 | 125.834 | 122.263 |
| 1,048,576 | 122.947 | 123.240 | 120.566 | 116.310 |

- **Interpretation**:
  - The catastrophic regression was largely an artifact of bench ordering and reading `out`; AVX2 recovered when run first and without reading `out` in the loop.
  - `Bind/avx2_ref` (pure loadu/xor/storeu) generally meets or beats header `Bind/avx2` at small/mid sizes, indicating minor header overhead (alignment branch) or aliasing.
  - For larger dims (≥64K), header `Bind/avx2` edges or matches core, with occasional wins (e.g., 65,536 bits).

- **Decision**:
  - Keep the conservative header path as default.
  - Consider adopting the reference pattern for small dims (no alignment checks) under a size threshold, pending a brief A/B.

- **Next**:
  - A/B-4 (loads alignment): force loadu in header path and compare vs current for small/mid dims.
  - If beneficial, introduce a size-based policy: use loadu-only for small dims, keep aligned loads for large dims.
  - Proceed to extend benchmarks to Hamming and permutation; document outcomes.

### 2025-10-06 04:18 — Permutation (Rotate) microbenchmark

- **Setup**:
  - Added `benchmarks/permute_bench.cpp` and CMake target `permute_bench`.
  - Benchmarks two variants for binary HVs:
    - `Permute/core_bitrotate`: current `core::PermuteRotate<Dim,bool>` (per-bit).
    - `Permute/word_rotate_ref`: bench-local word-wise rotate across 64-bit words with carry between words.

- **Results (GB/s, Release)**:

| Dim (bits) | core_bitrotate | word_rotate_ref |
|---:|---:|---:|
| 1,024 | 0.135 | 8.011 |
| 2,048 | 0.144 | 12.011 |
| 4,096 | 0.146 | 16.179 |
| 8,192 | 0.146 | 22.771 |
| 10,000 | 0.132 | 9.912 |
| 16,384 | 0.146 | 17.934 |
| 65,536 | 0.147 | 18.622 |
| 262,144 | 0.147 | 18.210 |
| 1,048,576 | 0.146 | 18.866 |

- **Interpretation**:
  - The per-bit implementation is 2–3 orders of magnitude slower than a word-wise rotate.
  - Word-level rotate sustains ~18–23 GB/s at large dims, consistent with memory traffic (read+write).

- **Action**:
  - Implement optimized `PermuteRotate<Dim,bool>` using 64-bit word rotation with a final-tail mask (same logic as `word_rotate_ref`).
  - Keep generic per-bit path for non-bool specializations; consider SIMD later if profiling justifies it.

### 2025-10-06 04:20 — Hamming microbenchmark results (scalar vs SSE2 vs AVX2)

- **Setup**: `benchmarks/hamming_bench.cpp` (read-only kernel). Measures GB/s for scalar `HammingDistance`, SSE2, AVX2 (Harley–Seal LUT+SAD).

- **Results (GB/s, Release)**:

| Dim (bits) | Hamming/core | Hamming/sse2 | Hamming/avx2 |
|---:|---:|---:|---:|
| 1,024 | 9.325 | 8.721 | 8.757 |
| 2,048 | 15.206 | 15.053 | 15.438 |
| 4,096 | 25.555 | 23.364 | 21.078 |
| 8,192 | 35.555 | 31.412 | 25.682 |
| 10,000 | 25.032 | 29.829 | 26.707 |
| 16,384 | 27.618 | 35.202 | 28.381 |
| 65,536 | 32.550 | 42.911 | 30.961 |
| 262,144 | 31.385 | 44.302 | 32.216 |
| 1,048,576 | 34.330 | 45.420 | 32.360 |

- **Interpretation**:
  - SSE2+POPCNT is strongest on this host for the read-heavy Hamming kernel, especially at large dims.
  - AVX2 (Harley–Seal) is competitive at small dims but not consistently faster; we will retain both backends and select per-size in future if warranted.

### 2025-10-06 04:26 — Permutation optimized: core bool rotates now word-wise

- **Change**: `include/hyperstream/core/ops.hpp` `PermuteRotate<Dim,bool>` now uses 64-bit word rotation with bit carry and final-word mask.
- **Benchmark**: re-ran `permute_bench`.

- **Results (GB/s, Release)**:

| Dim (bits) | Permute/core_bitrotate (optimized) | Permute/word_rotate_ref |
|---:|---:|---:|
| 1,024 | 6.750 | 6.195 |
| 2,048 | 11.849 | 11.940 |
| 4,096 | 18.250 | 15.801 |
| 8,192 | 21.524 | 21.981 |
| 10,000 | 9.449 | 9.489 |
| 16,384 | 26.586 | 17.750 |
| 65,536 | 27.215 | 18.278 |
| 262,144 | 27.489 | 18.654 |
| 1,048,576 | 28.031 | 18.831 |

- **Interpretation**:
  - The optimized core path now performs on par with or above the word-level reference and is near the memory-traffic ceiling at large dims.
  - This replaces the prior per-bit rotate that was orders of magnitude slower.

### 2025-10-06 04:00 — A/B-4 results: forced unaligned loads in header AVX2

- **Change**: In header `BindAVX2()`, forced `_mm256_loadu_si256` for inputs (removed alignment branch), kept `_mm256_storeu_si256` stores.

- **Results (GB/s, Release):**

| Dim (bits) | Bind/avx2 (loadu) | Bind/avx2_ref | Bind/core | Bind/sse2 |
|---:|---:|---:|---:|---:|
| 1,024 | 14.125 | 15.423 | 15.010 | 15.595 |
| 2,048 | 28.535 | 31.584 | 29.141 | 29.606 |
| 4,096 | 53.262 | 59.723 | 56.763 | 51.371 |
| 8,192 | 88.658 | 74.816 | 94.948 | 82.160 |
| 10,000 | 100.556 | 102.547 | 102.747 | 83.477 |
| 16,384 | 150.759 | 157.541 | 153.188 | 97.904 |
| 65,536 | 228.618 | 160.432 | 221.407 | 87.176 |
| 262,144 | 126.368 | 126.050 | 126.097 | 90.885 |
| 1,048,576 | 128.863 | 128.536 | 129.218 | 124.705 |

- **Interpretation**:
  - Forcing unaligned loads improved header AVX2 at small/mid sizes versus the alignment-branch variant, with no regressions at large sizes.
  - `avx2_ref` fluctuated slightly at 8K vs the earlier run (likely measurement variance), but overall trends hold.

- **Decision**:
  - Adopt loadu-only loads in `BindAVX2()` as the default (simpler and consistently good on this host).
  - Keep stores as storeu; keep NT store threshold env knob (default ≥1MB) for future A/B.

### 2025-10-06 04:45 — Associative memory microbenchmarks

- **Changes**:
  - Added `benchmarks/am_bench.cpp` to measure `PrototypeMemory<Dim,Capacity>::Classify()` throughput.
  - Added `benchmarks/cluster_bench.cpp` to measure `ClusterMemory::Update()/Finalize()` throughput.
  - Extended `PrototypeMemory` with an overload `Classify(query, dist_fn, default_label)` to inject a backend distance functor (SSE2/AVX2) for A/B; default remains `core::HammingDistance()` for portability.

- **Initial results (Release)**:
  - `AM/core,dim_bits=10000,capacity=256,size=256,iters=13863,secs=0.300012,queries_per_sec=46208.2,eff_gb_per_sec=14.916`
  - `AM/core,dim_bits=10000,capacity=256,size=256,iters=14129,secs=0.300011,queries_per_sec=47094.9,eff_gb_per_sec=15.202`

- **Interpretation**:
  - Core path delivers ~46–47k qps at 10k bits with 256 entries; consistent across runs. This meets the plan target (≥40k qps). SSE2/AVX2 distance functors to be A/B tested next.

- **Next**:
  - Run AM benches across capacities (256/1024/4096) and add SSE2/AVX2 distance functor results.
  - Add a concise “classify latency vs capacity” budget to `IMPLEMENTATION_PLAN.md`.



### 2025-10-06 04:52 — Core ops hardening + Bundler wide-mode toggle

- Changes
  - NormalizedHammingSimilarity is explicitly clamped to [-1, 1] to avoid FP excursions (include/hyperstream/core/ops.hpp).
  - BinaryBundler now uses int16_t saturating counters by default and documents rationale (literature aligns with narrow saturating counters; better cache locality).
  - Added compile-time escape hatch: define HYPERSTREAM_BUNDLER_COUNTER_WIDE to use int32_t counters and disable saturation.
    - Public alias: BinaryBundler<Dim>::counter_t; added static_asserts on size for both modes.
  - Tests:
    - Added saturation stress tests (±40k accumulations) to confirm behavior and majority rule ties.
    - Added wide-counter smoke test target compiled with the macro to verify counter_t is 32-bit.

- Build/test usage
  - Default (int16_t saturating):
    - ctest -C Release --output-on-failure
  - Wide counter smoke test is a separate target built with macro via target_compile_definitions:
    - Target: wide_counter_smoke_tests
    - Runs as part of ctest discovery; verifies sizeof(BinaryBundler<...>::counter_t) == 4.
  - Global opt-in (if you want to build other targets with wide counters):
    - GCC/Clang: cmake -S . -B build -DCMAKE_CXX_FLAGS="-DHYPERSTREAM_BUNDLER_COUNTER_WIDE"
    - MSVC:      cmake -S . -B build -DCMAKE_CXX_FLAGS="/DHYPERSTREAM_BUNDLER_COUNTER_WIDE"

- Rationale notes (recorded in code):
  - ±32,767 range per bit exceeds typical bundling needs; saturation prevents wrap-around while preserving majority semantics.
  - Hardware HDC work often uses 5–8 bit saturating counters or binarized bundling; int16 is a pragmatic, cache-friendly default.
  - Wide mode serves niche cases needing >32k undiminished accumulations without decay/chunking.


### 2025-10-06 05:02 — Phase 1: constexpr/noexcept hygiene for helpers

- Changes (include/hyperstream/core/ops.hpp)
  - detail::Popcount64: now `constexpr` and `noexcept` (bitwise-only, safe for compile-time; enables CTAD-like compile-time evaluation and inlining).
  - detail::InnerProductTerm<T> (arithmetic): now `constexpr` and `noexcept` for arithmetic types via SFINAE; complex overload unchanged (std::conj not constexpr in C++17).
  - detail::SquaredNorm<T> (arithmetic): now `constexpr` and `noexcept` for arithmetic types via SFINAE; complex overload unchanged for same reason.
- Rationale
  - Enables compile-time computation in constant contexts and encourages better inlining opportunities; preserves C++17 portability.
- Validation
  - Full test suite passed (28/28) in Release after changes.


### 2025-10-06 05:08 — Phase 1: constexpr/noexcept hygiene refinements (helpers)

- Scope: include/hyperstream/core/ops.hpp
- Changes
  - detail::Popcount64
    - Constrained to unsigned integral types via SFINAE to prevent accidental misuse
    - Remains `constexpr inline` and `noexcept` (bitwise-only; safe for constant expressions)
  - detail::InnerProductTerm
    - Arithmetic overload (for `std::is_arithmetic<T>`) remains `constexpr inline` and `noexcept`
    - Complex overload updated to `inline` + `noexcept` only (not `constexpr`) because `std::conj` and complex arithmetic are not `constexpr` in C++17
  - detail::SquaredNorm
    - Arithmetic overload remains `constexpr inline` and `noexcept`
    - Complex overload updated to `inline` + `noexcept` only for the same C++17 reason
- Rationale
  - Enables compile-time evaluation and better inlining for arithmetic cases while preserving strict C++17 portability for complex overloads
  - Constraining `Popcount64` avoids template instantiations on non-integral types
- Validation
  - Rebuilt in Release; full test suite passed 28/28 (includes wide-counter smoke test and saturation stress tests)


### 2025-10-06 05:12 — Phase 1: SIMD kernel agreement tests (backend vs core)

- Added test: `CpuBackend.AgreesWithCoreOnAwkwardDims` in `tests/backend_tests.cc`
- Dimensions covered: [1, 63, 64, 65, 100, 127, 128, 129, 10000]
  - Exercises single-bit, sub-word, exact word boundary, word+1, non-power-of-two, and large-scale cases
- Deterministic bit patterns:
  - a: `(i * 73 + 11) % 97 < 37`
  - b: `(i * 29 + 7) % 101 < 43`
- Verified operations:
  - Bind: `hyperstream::backend::Bind<Dim>` vs `hyperstream::core::Bind<Dim>` (bool specialization) produce identical word arrays
  - HammingDistance: `hyperstream::backend::HammingDistance<Dim>` equals `hyperstream::core::HammingDistance<Dim>`
- Notes:
  - Included `hyperstream/core/ops.hpp` and disambiguated existing tests to use backend-qualified calls where necessary
- Validation
  - Build: success
  - Tests: 29/29 passed (includes the new agreement test and the wide-counter smoke test)


### 2025-10-07 06:10 — Phase 1: Benchmark harness robustness + cluster default path (COMPLETE)

- Context
  - am_bench and cluster_bench occasionally showed no output or non-zero exit in direct console runs due to buffering/termination behavior; cluster_bench also crashed on Windows with a stack overflow in the default scenario.

- Changes
  - Both benchmarks
    - Added top-level function-try-block around main; return EXIT_SUCCESS on success; print error and return EXIT_FAILURE on exceptions.
    - Set stdout/stderr unbuffered at process start: setvbuf(stdout, nullptr, _IONBF, 0) and same for stderr to avoid environment-specific buffering quirks.
  - am_bench
    - No logic/timing changes; behavior unchanged aside from robust exit/prints.
  - cluster_bench
    - Root cause for crash: ClusterMemory<Dim,Capacity> was allocated on the stack with Capacity=1024 at Dim=10000 in the default path, overflowing the Windows console stack (0xC00000FD).
    - Fix: Default scenario now uses Capacity=16 and Updates=100 (still representative) and prints:
      - Immediate header line: "Cluster/default,dim_bits=10000,capacity=16,updates=100".
      - An "-update" metrics line after the first 150ms phase, then the final combined line after finalize.
    - Kept benchmark timing/measurement code intact; only default parameters, early prints, and robust exit handling were added.

- Validation (Release)
  - Direct run now produces metrics and exits 0:
    - Cluster/default,dim_bits=10000,capacity=16,updates=100
    - Cluster/update_finalize-update,dim_bits=10000,capacity=16,updates=100,update_iters=198,update_secs=0.150684,updates_per_sec=131401.0
    - Cluster/update_finalize,dim_bits=10000,capacity=16,updates=100,update_iters=198,update_secs=0.150684,updates_per_sec=131401.0,finalize_iters=5650,finalize_secs=0.150015,finalizes_per_sec=37662.9
  - am_bench prints metrics and returns 0.

- Status
  - Phase 1 TODO: "Fix benchmark exit codes and add default execution path for cluster_bench" — COMPLETE.


### 2025-10-07 — Phase 1: Adaptive Config + Capability Detection (C-lite) — COMPLETE
Context:
- Maintain 10k binary as default while enabling profile-driven defaults and runtime backend selection.
- Avoid large default stack allocations that caused cluster_bench overflow on Windows.

Changes:
- Added include/hyperstream/config.hpp: profiles (desktop default, embedded), defaults (dim=10000/2048, cap=256/16), heap threshold, helpers.
- Added include/hyperstream/backend/capability.hpp: x86/x64 CPUID probing (SSE2, AVX2) with safe fallbacks, feature mask API.
- Added include/hyperstream/backend/policy.hpp: runtime backend selection with scalar/SSE2/AVX2, compile-time escape via HYPERSTREAM_FORCE_SCALAR.
- Updated include/hyperstream/memory/associative.hpp: move ClusterMemory sums_ to heap (unique_ptr<int[]>); PrototypeMemory entries_ to heap (unique_ptr<Entry[]>). Preserves API, prevents stack overflow across profiles.
- New tests: config_tests, capability_tests, policy_tests (API, detection self-consistency, backend agreement with core).
- New benchmark: benchmarks/config_bench.cpp prints profile/defaults/features/selected backends.
- Updated am_bench and cluster_bench to print configuration summary at startup.

Validation:
- Build: MSVC Release OK; strict warnings.
- Tests: 32/32 passing (existing 29 + 3 new).
- Benchmarks: config_bench, am_bench, cluster_bench print config and exit 0.
  - config_bench sample: profile=desktop, mask shows SSE2+AVX2, selected backends avx2.
  - Performance: AM/Cluster within noise vs prior runs (<=5%).

Notes/Trade-offs:
- Heap allocation is used unconditionally for large memory classes to keep API stable and reduce fragile size heuristics; policy header documents thresholds for future refinement.
- Capability detection guards non-x86 with safe false; FORCE_SCALAR disables runtime checks.

Status:
- Acceptance criteria met: compiles cleanly, tests passing, benchmarks robust, no regressions observed.


### 2025-10-07 — Phase 1: SIMD dispatch policy refinement + footprint reporting — COMPLETE
Context:
- Host benches show SSE2+POPCNT wins Hamming at large dims on this machine. Add a minimal heuristic while preserving AVX2 for bind and smaller dims.

Changes:
- Policy: include/hyperstream/backend/policy.hpp
  - Added BackendKind enum, GetBackendName(), and PolicyReport Report<Dim>() API.
  - Added kHammingPreferSSE2DimThreshold=16384; Hamming chooses SSE2 when Dim ≥ threshold and AVX2 is otherwise available; bind remains AVX2-first.
  - Added reasons strings for transparency.
- Config: include/hyperstream/config.hpp
  - Added constexpr storage footprint helpers for BinaryHyperVector, PrototypeMemory, ClusterMemory, CleanupMemory.
- Benchmarks: benchmarks/config_bench.cpp prints backend selection with rationale and footprint estimates.
- Tests: extended policy_tests and config_tests for heuristics and footprint helpers.

Validation:
- Build: MSVC Release OK; strict warnings.
- Tests: 34/34 passing (added 2 tests).
- config_bench sample output:
  - SelectedBackends: bind=avx2 ("wider vectors"), hamming=avx2 at Dim=10000 (< threshold).
  - Footprints: Binary10k=1256B; Cluster<10k,16>=640,192B; Prototype<10k,256>=323,584B.

Notes:
- Heuristic threshold is conservative and host-informed; future work can refine with microbench auto-tuning or env override.
- Footprint helpers report storage-only and are constexpr; exclude control fields.


### 2025-10-07 — Phase 1: Env override + Auto-tune microbench + Docs — COMPLETE
Context:
- Make Hamming SSE2/AVX2 threshold adjustable at runtime; provide a quick auto-tune; document new policy and footprint helpers.

Changes:
- Policy (include/hyperstream/backend/policy.hpp):
  - Added GetHammingThreshold() reading HYPERSTREAM_HAMMING_SSE2_THRESHOLD; HammingThresholdOverridden().
  - DecideHamming() now respects the active (env-aware) threshold.
  - Doxygen comments for BackendKind, GetBackendName(), PolicyReport, Report<Dim>(), threshold APIs.
- Config (include/hyperstream/config.hpp):
  - Doxygen comments for footprint helpers.
- Benchmarks (benchmarks/config_bench.cpp):
  - Prints Policy/HammingThreshold and overridden flag.
  - Added --auto-tune: microbench SSE2 vs AVX2 at dims {8k,16k,32k,64k}, <2s runtime; prints per-dim timings and a recommended threshold.
- Tests:
  - policy_tests: added ThresholdDefaultWhenEnvUnset.
- Docs:
  - Added docs/Configuration.md describing profiles, policy, env overrides, Report<Dim>(), and footprint helpers.

Validation:
- Build: MSVC Release OK; strict warnings.
- Tests: 35/35 passing.
- config_bench:
  - Default run shows Policy/HammingThreshold=16384 (overridden=0).
  - Auto-tune example showed SSE2 faster starting at 8192 on this host; recommended_threshold=8192; configured remains 16384.

Notes:
- Auto-tune uses deterministic patterns and fixed iteration budgets per dimension to keep noise and runtime low.
- The recommended threshold is informational; users can set HYPERSTREAM_HAMMING_SSE2_THRESHOLD to adopt it without rebuild.

### 2025-10-07 — Phase 2: Runtime dispatch (Selection → Expansion → Simulation) — COMPLETED

- Implementation
  - GCC/Clang: function-level target attributes for raw SSE2/AVX2 kernels (no global -m flags)
  - MSVC: separate translation units compiled with /arch:SSE2 and /arch:AVX2; aggregated as hyperstream_kernels
  - Transitive linkage cleanup: tests/benchmarks now rely on INTERFACE propagation from `hyperstream`
  - Policy: runtime-only decisions based on feature mask and dimension heuristic; compile-time guards removed from selection logic
  - Tests: dispatch selection, no-illegal-instruction guards, env override, forced-scalar target (per-target definition)
  - Documentation: added Docs/Runtime_Dispatch.md (architecture, adding ISAs, invariants, CI)
- Validation
  - MSVC Release: all 39 tests passed, including new dispatch and forced-scalar tests
  - Ready for CI validation on Linux/macOS (no global -mavx2; function attributes localize ISA code paths)

### 2025-10-07 — Phase 2 integrated to main + CI hardening — COMPLETE

- Context
  - PR #8 (feat/phase2-runtime-dispatch-sim-docs) merged to main; feature branch deleted. Phase 2 runtime dispatch is fully integrated.

- Changes
  - CI: Policy/Capability checks and Force Scalar stabilized across ubuntu/windows/macos.
    - Fixed brittle grep patterns in SelectedBackends assertions (match `hamming=...` on the same line as `bind=...`).
    - Autotune step now accepts both numeric thresholds and `(none within tested range)` when AVX2 dominates.
    - Introduced CMake option `-DHYPERSTREAM_FORCE_SCALAR=ON` (replaced raw `CMAKE_CXX_FLAGS` defines) to avoid MSYS path-mangling on Windows.
  - Code: No functional changes to dispatch/backends beyond prior platform guards and policy; product logic unchanged.
  - Docs: Runtime dispatch architecture is covered in Docs/Runtime_Dispatch.md; README and CONTRIBUTING reference it.

- Validation
  - All CI checks passed on ubuntu-latest, windows-latest, macos-latest.
  - Jobs green: Build+Test, Policy/Capability Checks, Force Scalar.

- Notes
  - Output format from `benchmarks/config_bench` (profile, CPUFeatures, Policy/HammingThreshold, SelectedBackends) is now asserted in CI with robust patterns.
  - Windows configure uses VS2022 generator with `-DHYPERSTREAM_FORCE_SCALAR=ON` for the force-scalar job.

- Next (Phase 3/4 closeout per high-function workflow)
  - Phase 3: Testing/CI
    - Deterministic RNG in `core_ops_extended_tests` (fixed seed).
    - Add tests: default_label on empty `PrototypeMemory`; Capacity=0 safety for `PrototypeMemory`/`CleanupMemory`.
    - Property-based tests with fixed seeds: algebraic properties and backend equality.
    - CI: add build matrix with portable (runtime-dispatch) and AVX2-optimized test runners.
  - Phase 4: Docs/Benchmarks
    - Re-enable/harden benches on all platforms (scalar-only on non-x86); add warmup + multi-sample stats; JSON/NDJSON output.
    - README: Design/Performance (runtime dispatch, unaligned SIMD IO, tail-bit masking).
    - Docs: associative memory capacity/eviction/thread-safety; note unaligned load/store assumption in SSE2/AVX2 headers.

### 2025-10-07 — Phase 3: PrototypeMemory default_label test — COMPLETE

- Context
  - Phase 3 Testing/CI: add deterministic, edge-case tests without changing product behavior.
- Changes
  - tests/associative_tests.cc: added `PrototypeMemory.ClassifyReturnsDefaultLabelWhenEmpty`.
    - Constructs empty PrototypeMemory and verifies `Classify(query, default_label)` returns the provided default when size()==0.
- Validation
  - Built associative_tests target and ran the single test: passed.
  - Full suite (Release, MSVC): 43/43 tests passed (1 disabled placeholder).
- Status
  - Task complete; proceed to Capacity=0 behavior tests next.

### 2025-10-07 — Phase 3: Capacity=0 behavior tests (Prototype/Cleanup) — COMPLETE

- Context
  - Phase 3 Testing/CI: validate behavior at zero capacity; no product code changes.
- Changes
  - tests/associative_tests.cc:
    - PrototypeMemory.ZeroCapacityBehavior: Learn() fails; Classify() returns default_label when size()==0.
    - CleanupMemory.ZeroCapacityBehavior: Insert() fails; Restore() returns fallback when size()==0.
- Validation
  - Built associative_tests and ran the two tests in isolation: both passed.
  - Full suite (Release, MSVC): 45/45 tests passed (1 disabled placeholder).
- Status
  - Task complete; next up: property-based tests with fixed seeds.


### 2025-10-07 — Phase 3: Property-based tests (fixed seeds) — COMPLETE

- Context
  - Phase 3 Testing/CI: add property-based tests with deterministic RNG; validate backend equality.
- Changes
  - tests/core_ops_tests.cc:
    - PropertyCoreOps.BindInvertibility_FixedSeed: bind(bind(a,key),key)==a for random a,key (mt19937(42)).
    - PropertyCoreOps.HammingTriangleInequality_FixedSeed: d(a,c) <= d(a,b)+d(b,c) for random a,b,c.
  - tests/backend_tests.cc:
    - CpuBackend.AgreesWithCoreOnRandomVectors_FixedSeed: backend Bind/Hamming equal to core across dims {64,256,1000}, 40 iters, mt19937(42).
- Validation
  - Target runs: new tests passed in isolation.
  - Full suite (Release, MSVC): 48/48 tests passed (1 disabled placeholder).
- Status
  - Task complete; Phase 3 remains IN_PROGRESS.


### 2025-10-07 — Phase 3: CI matrix update — AVX2 preference job added — COMPLETE

- Context
  - Phase 3 Testing/CI: explicitly assert AVX2 backend selection on x86 GitHub runners.
- Changes
  - .github/workflows/ci.yml: added `avx2-check` job (matrix: ubuntu-latest, windows-latest).
    - Configure Release; build; run `benchmarks/config_bench`.
    - Assert via grep: `SelectedBackends/bind=avx2` and `hamming=avx2` at dim=10000.
    - macOS excluded (ARM).
- Validation
  - Local: config_bench emits lines matched by grep; policy threshold (16384) implies AVX2 for Hamming at 10k.
  - CI: existing jobs unchanged and expected to remain green; avx2-check runs on x86 runners.
- Status
  - Task complete; Phase 3 remains IN_PROGRESS.


### 2025-10-07 — Phase 4: README Design/Performance section — COMPLETE

- Context
  - Add user-facing documentation summarizing runtime dispatch, unaligned IO invariants, tail handling, and performance guidelines; link to Docs/Runtime_Dispatch.md.
- Changes
  - README.md: new "Design and Performance" section after Runtime dispatch architecture.
- Validation
  - Markdown renders cleanly; no build/test changes.
- Status
  - Task complete; Phase 4 remains IN_PROGRESS.

### 2025-10-07 — Phase 4: Backend/Associative docs + Bench warmup/NDJSON — COMPLETE

- Context
  - Phase 4 tasks A/B/C: document SIMD backend invariants; document associative memories; enhance benchmarks with warmup, multi-sample, and NDJSON.

- Changes
  - Backend docs (headers):
    - include/hyperstream/backend/cpu_backend_sse2.hpp and cpu_backend_avx2.hpp
      - Added top-of-file invariant blocks: unaligned load/store semantics (loadu/storeu), contiguous uint64_t layout via HyperVector::Words(), safe tail handling via final-word mask, compiler target attributes policy (GCC/Clang function-level; MSVC in .cpp).
      - Added Doxygen for BindWords()/HammingWords() parameters and semantics.
  - Associative memory docs:
    - include/hyperstream/memory/associative.hpp: class-level Doxygen for PrototypeMemory, ClusterMemory, CleanupMemory covering: fixed Capacity with no eviction, size()==0 behavior, Capacity==0 behavior, not thread-safe (external synchronization required), and complexity for Learn/Update/Classify/Finalize.
  - Benchmarks:
    - benchmarks/am_bench.cpp and benchmarks/cluster_bench.cpp
      - Added CLI flags: --warmup_ms, --measure_ms, --samples, --json.
      - Implemented warmup phase (optional) before measurement; default warmup_ms=0 to preserve legacy behavior.
      - Multi-sample runs with mean/median/stdev aggregate; NDJSON mode emits one JSON object per sample (+ optional aggregate).
      - Backward compatibility: when flags omitted, output format and timings remain unchanged; default measure_ms matches previous (AM=300ms, Cluster=150ms).

- Validation
  - Local MSVC Release build OK; no new warnings.
  - am_bench: verified CSV default output unchanged; with --json --samples=3, NDJSON lines parse and include expected fields.
  - cluster_bench: default path preserved (config + default line + metrics); NDJSON validated similarly.

- Status
  - Phase 4 tasks A/B/C: COMPLETE.




### 2025-10-08 — Phase A: NEON integration + CI perf regression stabilization — COMPLETE

- Context
  - Phase A prioritized CI performance regression infrastructure; we pulled forward ARM NEON support to make macOS arm64 first‑class in both tests and benchmarks.
  - Several CI rounds exposed macOS build/test gaps and Windows perf variance; we iterated safely to green.

- Changes
  - Backend (AArch64):
    - Added header‑only NEON backend: include/hyperstream/backend/cpu_backend_neon.hpp (Bind/Hamming with vld1q/veorq/vcntq/vaddvq).
    - Capability: extended CpuFeature with NEON; detection on ARM64.
    - Policy: on ARM64, ignore synthetic x86 feature masks in selection and use host features directly; map to NEON where available.
  - Tests:
    - Dispatch tests updated to expect NEON on ARM64 instead of scalar; property test now includes NEON header under HS_ARM64_ARCH.
  - Benchmarks (arm64 enablement):
    - Guarded x86‑only includes and lanes under HS_X86_ARCH; added NEON lanes where applicable.
    - Fixed config_bench brace/guard structure; auto‑tune and microbench blocks compile cleanly on macOS arm64.
  - CI (perf regression, Windows variance):
    - Perf Regression workflow raises Windows tolerances to 25% (QPS/GBPS) to absorb runner variance pending more samples.

- Validation
  - Local (Windows, Release): full suite passed prior to push.
  - CI (latest commit cf0c1fa, PR #27):
    - CI / Build and Test (macos‑latest): PASS (dispatch tests green with NEON expectations).
    - CI / Build and Test (ubuntu‑latest, windows‑latest): PASS.
    - Perf Regression (NDJSON) — ubuntu/macos/windows: PASS; Windows passed with 25% tolerance step.

- Notable Fixes across iterations
  - macOS arm64 build: added arch guards to hamming_bench/config_bench; introduced NEON hamming lane.
  - macOS arm64 compile error: corrected preprocessor/brace structure in config_bench.
  - ARM64 dispatch: first sanitized mask → scalar fallback on synthetic x86 masks (rejected); final approach ignores synthetic mask and uses host features.
  - Tests: property test updated to expect NEON on ARM64; now consistent with policy and backend availability.

- Decisions
  - Treat NEON as baseline on ARM64; tests and policy reflect this.
  - For Windows perf CI, prefer tolerance widening (25%) over constantly re‑basing, until we have multiple samples to set robust medians.

- Next Steps
  - Capture macOS arm64 NDJSON aggregates from multiple runs; update ci/perf_baseline/macos/*.json from means/medians and consider narrowing tolerances.
  - Document NEON specifics (alignment semantics, popcount approach, tail handling) in a brief Docs/NEON_Notes.md.
  - Observe Windows variance over 2–3 runs; tighten tolerances if stable.

- Evidence
  - CI run (CI): 18333229927 — macOS Build and Test job success.
  - CI run (Perf Regression): 18333229963 — Windows/Ubuntu/macOS perf jobs success; Windows tolerance step executed.

### 2025-10-08 — Phase B: Encoders + Serialization — COMPLETE

- Context
  - Elevate HyperStream from core ops + associative memories to production-grade data representations and persistence.
  - Ensure deterministic, seed-stable encodings and minimal, forward-compatible binary I/O without external deps.

- Deliverables (merged via PR #28)
  - Encoders (header-only, zero deps)
    - `ItemMemory<Dim>`: deterministic symbol→HV mapping (SplitMix64 PRNG; trailing-bit masking)
    - `SymbolEncoder<Dim>`: role-based rotation wrapper around ItemMemory (composition-friendly)
    - `ThermometerEncoder<Dim>`: scalar→dense monotonic code; controllable resolution
    - `RandomProjectionEncoder<Dim>`: numeric vector→binary HV via seeded projections; deterministic
  - Serialization (HSER1)
    - Header-only binary save/load for PrototypeMemory and ClusterMemory (little-endian; size checks; no exceptions; bool-return)
    - Minimal read-only accessors added to associative.hpp (const data/view and LoadRaw)
  - Tests
    - Determinism, density, role-rotation, projection stability, round-trip I/O, corruption/mismatch detection
    - Full suite passing (61/61 before merge); zero new warnings under strict flags
  - Docs
    - Docs/Encoders.md and Docs/Serialization.md (format spec, invariants, examples, threading notes)

- CI/CD — macOS perf baseline alignment (PR #29)
  - Issue: Baselines captured on macOS‑15/AppleClang 17; CI pinned to macOS‑14/AppleClang 15 → apparent ~30% AM regressions
  - Fixes:
    - Workflow pinned perf job to macos-14; normalized OS_NAME → baseline folder mapping (linux/macos/windows)
    - Re-captured baselines from fresh macOS‑14 run; stored medians in ci/perf_baseline/macos/*.json
  - Outcome: Perf Regression (NDJSON) macOS job passes at existing 25% tolerances; Ubuntu/Windows unaffected

- Key technical decisions
  - Deterministic encoders default to fixed seeds; role-based permutations use word-rotate with tail masking
  - Serialization favors simplicity and robustness: fixed magic ("HSER1"), explicit sizes, strict little‑endian, no exceptions
  - Keep macOS/Windows perf tolerances at 25% pending multi-run variance analysis; Linux remains tighter

- Lessons learned
  - CI runner provenance matters: OS/compilers can shift medians by >25%; baselines must match runner family
  - Normalize workflow OS naming for baseline lookup to avoid path drift on matrix changes
  - Prefer medians over means for stable baselines; enforce warmup + multiple samples in benches

- Follow-ups
  - After several stable macOS‑14 runs, consider narrowing tolerances; reassess macOS‑15 (macos-latest) when deliberately re-baselining
  - Add cross-platform golden-file checks for serialization (byte‑exactness) and encoder determinism across compilers

### 2025-10-08 — Phase C: Golden serialization scaffolding + encoder determinism + golden-compat (read-only)

- PR #32 opened: golden serialization idempotence tests + encoder determinism tests; new Golden Compatibility CI job.
- Windows build fixed by switching to `cmake --build ... --parallel 2` (commit 5b3afeb); MSBuild no longer sees `-j`.
- Outcome: Golden Compatibility PASS on ubuntu/windows/macos-14; macOS-14 Perf Regression also PASS on this PR.
- Next: Commit canonical golden byte fixtures with SHA-256 manifest and make tests assert fixtures; then add canonical per-encoder hashes across compilers.

### 2025-10-09 — Phase C: Canonical per-encoder hashes validated in CI; backend parity task kicked off

- Context
  - Captured canonical 64-bit FNV-1a hashes for ItemMemory, SymbolEncoder (role rotations), ThermometerEncoder, RandomProjectionEncoder at D=256; added D=1024 dump helper.
  - Committed tests/golden/encoder_hashes.json and extended tests/encoder_determinism_tests.cc to assert against it.
  - Opened PR #33 to trigger Golden Compatibility across platforms for commits 9620a30, 96e9867, 8c9d5eb.

- CI Validation (PR #33)
  - Golden Compatibility: PASS on all platforms for head 8c9d5eb
    - ubuntu-latest: success
    - windows-latest: success
    - macos-14: success

- Artifacts
  - tests/golden/encoder_hashes.json: canonical per-encoder hashes (windows-msvc/linux-gcc/macos-clang entries are equal by design)
  - Docs/Reproducibility.md: determinism guarantees and regeneration steps

- Next task (approved): Backend parity + provenance (tests/CI/docs-only)
  - Add a second build/run in Golden Compatibility with -DHYPERSTREAM_FORCE_SCALAR=ON; hashes must match default (SIMD) build.
  - Augment provenance: print compiler versions and CMake cache entries (CMAKE_CXX_COMPILER, CMAKE_CXX_FLAGS, CMAKE_BUILD_TYPE, HYPERSTREAM_FORCE_SCALAR) for both builds.
  - Tests annotate backend mode via RecordProperty("backend", "simd" | "scalar").
  - Update Docs/Reproducibility.md with backend parity and provenance verification.


### 2025-10-09 — Phase C: Backend parity + provenance — COMPLETE

- PR #33 merged to main; feature branch deleted.
- Commits included: 9620a30, 96e9867, 8c9d5eb, 26b4068, 5b051b1.
- Backend parity:
  - Golden Compatibility now runs determinism/golden tests twice: default (SIMD) and force-scalar (`-DHYPERSTREAM_FORCE_SCALAR=ON`).
  - Encoder hashes must match in both modes; tests annotate `backend` via `RecordProperty` for provenance.
- Provenance enhancements:
  - Workflow logs compiler versions and key CMake cache entries for both builds: `CMAKE_CXX_COMPILER`, `CMAKE_CXX_FLAGS`, `CMAKE_BUILD_TYPE`, `HYPERSTREAM_FORCE_SCALAR`.
- Windows perf baseline parity:
  - Perf Regression matrix pinned to `windows-2022` to match committed baseline environment and avoid false regressions introduced by `windows-latest` (Windows Server 2025) image shift.
- CI results: All green — Golden Compatibility, Perf Regression (ubuntu-latest, windows-2022, macos-14), and standard CI.


### 2025-10-09 — Phase D: Perf CI robustness & baseline hygiene — KICKOFF

- Branch: feat/phase-d-perf-ci-robustness-20251009
- Scope: tests/scripts/docs-only; zero library code changes; zero external deps.
- Validator (scripts/bench_check.py):
  - Enforce variance-bounds (stdev/mean) per OS when samples ≥ 3: Linux ≤0.10, macOS ≤0.20, Windows ≤0.25
  - Emit per-group aggregates with provenance to perf_agg.ndjson (runner_os, image_os, image_version, compiler)
- Workflow (.github/workflows/perf-regression.yml):
  - Added provenance echo step and artifact upload of perf_agg.ndjson
- Docs: Reproducibility.md updated with thresholds and windows-2022 baseline refresh procedure
- Next: Open PR and validate CI across ubuntu-latest, windows-2022, macos-14


### 2025-10-10 — Phase E: Temporal/Streaming determinism — KICKOFF

- Branch: feat/phase-e-streaming-determinism-20251010
- Scope: tests/CI/docs-first; zero external deps; no library changes unless tests surface an ambiguity that requires a minimal, backward-compatible hook
- Goals:
  - Prove chunking/interleave invariance for a canonical event stream and seed
  - Parity across OS/compilers and SIMD vs scalar
  - Golden fixtures (events NDJSON) + reference hashes; CI coverage in golden-compat (both default and force-scalar)
- Deliverables:
  - tests/streaming_determinism_tests.cc, tests/golden/streaming_events.ndjson, tests/golden/streaming_hashes.json
  - Docs: Reproducibility.md (Streaming determinism) and optional Docs/Temporal.md
  - CI: extend golden-compat to include streaming determinism with provenance
