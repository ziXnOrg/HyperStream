# HyperStream

High-performance, header-only C++ library for Hyperdimensional Computing (HDC). HyperStream provides binary hypervectors, SIMD-accelerated kernels (SSE2/AVX2), adaptive runtime backend selection, associative memory data structures, and a streaming classification framework.

- Header-only, C++17
- Cross-platform: Windows, Linux, macOS
- Runtime capability detection and policy-driven backend selection
- Safe-by-default configurations with embedded/desktop profiles

## Getting started

### Requirements
- CMake 3.16+
- A C++17 compiler
  - MSVC 2022, Clang 12+, or GCC 10+

### Build and test

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j 4
ctest --test-dir build -C Release --output-on-failure
```

### Quick config report

```text
# Print profile, CPU features, selected backends, and footprint estimates
./build/benchmarks/config_bench        # Unix
./build/benchmarks/Release/config_bench.exe  # Windows
```

Optional: quick auto-tuning of Hamming backend threshold for your host.

```text
./build/benchmarks/config_bench --auto-tune
```

## Core concepts

- Binary hypervectors with bit-packed storage
- Operations: bind (XOR), bundle (majority), permute (rotate), similarity (Hamming)
- SIMD backends for bind and Hamming
- Policy: runtime CPU detection + dimension-based heuristics

## Policy and capability

HyperStream chooses backends at runtime based on the CPU and simple heuristics. You can query decisions via the policy report API.

```cpp
#include "hyperstream/backend/policy.hpp"

constexpr std::size_t D = 10000;
auto rep = hyperstream::backend::Report<D>();
// rep.bind_kind, rep.hamming_kind, and reason strings
```

### Environment overrides

- Adjust Hamming SSE2 preference threshold (default: 16384) without rebuilding:

```text
export HYPERSTREAM_HAMMING_SSE2_THRESHOLD=32768  # Unix
set HYPERSTREAM_HAMMING_SSE2_THRESHOLD=32768     # Windows
```

- Force scalar backends at compile time (e.g., portability checks):

```text
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-DHYPERSTREAM_FORCE_SCALAR"
```

## Runtime dispatch architecture

HyperStream builds universal binaries and selects SIMD backends at runtime to avoid illegal-instruction traps while still delivering optimized SSE2/AVX2 performance.

See Runtime Dispatch Architecture (Docs/Runtime_Dispatch.md) for details on SIMD backend selection and adding new ISA support.

## Memory footprint helpers

Constexpr helpers estimate storage requirements for common structures.

```cpp
#include "hyperstream/config.hpp"
using namespace hyperstream::config;

constexpr std::size_t d = 10000;
auto hv_bytes = BinaryHyperVectorStorageBytes(d);
```

## Benchmarks

- config_bench: configuration, capability, and policy report; optional `--auto-tune`
- am_bench: associative memory microbenchmark
- cluster_bench: clustering microbenchmark

```text
./build/benchmarks/config_bench --auto-tune
./build/benchmarks/am_bench
./build/benchmarks/cluster_bench
```

## Project status

- Policy supports SSE2/AVX2 backends with runtime selection
- Env-var threshold override for Hamming is available; auto-tuning provides recommendations
- Tests validate correctness across awkward dimensions and backends

## Contributing

Issues and pull requests are welcome. Please run the full test suite and ensure Release builds are clean with strict warnings.

## CI

This repository is set up with GitHub Actions to build on Windows, Linux, and macOS, run tests, validate policy output, and verify scalar fallback.

[![CI](https://github.com/ziXnOrg/HyperStream/actions/workflows/ci.yml/badge.svg)](https://github.com/ziXnOrg/HyperStream/actions/workflows/ci.yml)
