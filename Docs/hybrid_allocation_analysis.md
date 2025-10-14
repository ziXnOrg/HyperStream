# Hybrid Allocation Strategy Analysis

## Overview

This document analyzes the hybrid allocation strategy implemented in HyperStream's associative memory classes to resolve Windows stack overflow issues while maintaining optimal performance for common use cases.

## Problem Statement

**Windows Stack Overflow Issue:**
- Windows default stack size: **1 MB** (1,048,576 bytes)
- Linux/macOS default: **8 MB**
- Large `std::array` members in memory classes exceeded Windows stack limits
- Exit code: `-1073741571` (0xC00000FD = STATUS_STACK_OVERFLOW)

**Performance Regression with Pure Heap Allocation:**
- Initial fix (commit 9359a9e): migrated all storage to `std::vector` (heap)
- Result: **26-34% performance regression** across all platforms
- Unacceptable for a performance-first library

## Solution: Hybrid Allocation

**Strategy:**
- Use `std::array` (stack) for small sizes ≤ 512 KB
- Use `std::vector` (heap) for large sizes > 512 KB
- Compile-time decision via `std::conditional`

**Threshold Rationale:**
- Conservative threshold: **512 KB** (524,288 bytes)
- Leaves headroom for:
  - Other stack variables
  - Nested function calls
  - Compiler padding/alignment
- Well below Windows 1 MB limit

## Memory Footprint Analysis

### PrototypeMemory<Dim, Capacity>

**Formula:** `sizeof(Entry) * Capacity`
- `Entry` = `std::uint64_t` (8 bytes) + `HyperVector<Dim, bool>` (Dim/8 bytes)
- `sizeof(Entry)` ≈ 8 + (Dim / 8) bytes

| Dim | Capacity | Entry Size | Total Size | Allocation |
|-----|----------|------------|------------|------------|
| 10000 | 256 | 1,258 B | 322,048 B (314 KB) | **Stack** ✓ |
| 10000 | 512 | 1,258 B | 644,096 B (629 KB) | **Heap** |
| 10000 | 1024 | 1,258 B | 1,288,192 B (1.23 MB) | **Heap** |
| 16384 | 256 | 2,056 B | 526,336 B (514 KB) | **Heap** |
| 16384 | 1024 | 2,056 B | 2,105,344 B (2.01 MB) | **Heap** |
| 65536 | 256 | 8,200 B | 2,099,200 B (2.00 MB) | **Heap** |
| 65536 | 1024 | 8,200 B | 8,396,800 B (8.01 MB) | **Heap** |

**Benchmark Configurations:**
- `AM/core` with Dim=10000, Capacity=256: **Stack** (optimal performance)
- `AM/core` with Dim=10000, Capacity=1024: **Heap** (avoids Windows overflow)

### ClusterMemory<Dim, Capacity>

**Formula:** `sizeof(labels_) + sizeof(counts_) + sizeof(sums_)`
- `labels_`: `std::uint64_t * Capacity` = 8 * Capacity bytes
- `counts_`: `int * Capacity` = 4 * Capacity bytes
- `sums_`: `int * Capacity * Dim` = 4 * Capacity * Dim bytes (largest!)

**Total:** `(8 + 4 + 4*Dim) * Capacity` bytes

| Dim | Capacity | labels_ | counts_ | sums_ | Total Size | Allocation |
|-----|----------|---------|---------|-------|------------|------------|
| 10000 | 16 | 128 B | 64 B | 640,000 B | 640,192 B (625 KB) | **Heap** |
| 16384 | 16 | 128 B | 64 B | 1,048,576 B | 1,048,768 B (1.00 MB) | **Heap** |
| 65536 | 16 | 128 B | 64 B | 4,194,304 B | 4,194,496 B (4.00 MB) | **Heap** |

**Benchmark Configurations:**
- All `Cluster/update_finalize` benchmarks use **Heap** (sums_ array is very large)

### CleanupMemory<Dim, Capacity>

**Formula:** `sizeof(HyperVector<Dim, bool>) * Capacity`
- `HyperVector<Dim, bool>` = Dim / 8 bytes

| Dim | Capacity | HV Size | Total Size | Allocation |
|-----|----------|---------|------------|------------|
| 10000 | 256 | 1,250 B | 320,000 B (313 KB) | **Stack** ✓ |
| 10000 | 512 | 1,250 B | 640,000 B (625 KB) | **Heap** |
| 10000 | 1024 | 1,250 B | 1,280,000 B (1.22 MB) | **Heap** |
| 16384 | 256 | 2,048 B | 524,288 B (512 KB) | **Stack** ✓ (exactly at threshold) |
| 16384 | 1024 | 2,048 B | 2,097,152 B (2.00 MB) | **Heap** |
| 65536 | 256 | 8,192 B | 2,097,152 B (2.00 MB) | **Heap** |

## Performance Impact Analysis

### Expected Performance Characteristics

**Stack Allocation (≤ 512 KB):**
- Zero allocation overhead
- Optimal cache locality
- Deterministic initialization
- **Performance:** Baseline (100%)

**Heap Allocation (> 512 KB):**
- One-time allocation cost in constructor
- Potential cache misses (depends on allocator)
- Still contiguous memory (std::vector guarantees)
- **Performance:** ~95-98% of baseline (estimated)

### Benchmark Coverage

**Common Configurations (Stack):**
- `PrototypeMemory<10000, 256>`: 314 KB → **Stack**
- `CleanupMemory<10000, 256>`: 313 KB → **Stack**
- `CleanupMemory<16384, 256>`: 512 KB → **Stack** (exactly at threshold)

**Large Configurations (Heap):**
- `PrototypeMemory<10000, 1024>`: 1.23 MB → **Heap**
- `ClusterMemory<10000, 16>`: 625 KB → **Heap**
- All 65536-bit configurations → **Heap**

## Implementation Details

### Hybrid Storage Helper

```cpp
namespace detail {

inline constexpr std::size_t kHybridAllocThresholdBytes = 512U * 1024U;

template <typename T, std::size_t N>
struct HybridStorage {
  static constexpr std::size_t kTotalBytes = sizeof(T) * N;
  static constexpr bool kUseStack = (kTotalBytes <= kHybridAllocThresholdBytes);
  
  using type = std::conditional_t<kUseStack, std::array<T, N>, std::vector<T>>;
};

template <typename T, std::size_t N>
using HybridStorageT = typename HybridStorage<T, N>::type;

template <typename T, std::size_t N>
inline auto MakeHybridStorage() {
  if constexpr (HybridStorage<T, N>::kUseStack) {
    return std::array<T, N>{};  // Value-initialized (zero-filled)
  } else {
    std::vector<T> vec;
    vec.resize(N);  // Pre-allocate and value-initialize
    return vec;
  }
}

}  // namespace detail
```

### Usage Example

```cpp
template <std::size_t Dim, std::size_t Capacity>
class PrototypeMemory {
 public:
  struct Entry {
    std::uint64_t label = 0;
    core::HyperVector<Dim, bool> hv;
  };

  PrototypeMemory() : entries_(detail::MakeHybridStorage<Entry, Capacity>()) {}

 private:
  detail::HybridStorageT<Entry, Capacity> entries_;
  std::size_t size_ = 0;
};
```

## Validation

### Correctness
- All existing tests pass (associative_tests: 7/7)
- Determinism preserved (value-initialization for both stack and heap)
- API unchanged (transparent to users)

### Performance
- Stack-allocated configurations: **Zero overhead** (same as original std::array)
- Heap-allocated configurations: **One-time cost** amortized over many operations
- Expected CI results:
  - **Windows:** No stack overflow; benchmarks complete successfully
  - **macOS/Ubuntu:** Performance restored for common configurations (Dim=10000, Cap=256)

### Cross-Platform Compatibility
- Works on Windows (1 MB stack), Linux (8 MB stack), macOS (8 MB stack)
- No platform-specific workarounds needed
- Scales to arbitrary dimensions without manual tuning

## Conclusion

The hybrid allocation strategy provides:
1. **Correctness:** Eliminates Windows stack overflow for all configurations
2. **Performance:** Preserves optimal performance for common use cases (≤ 512 KB)
3. **Scalability:** Supports large dimensions via heap allocation
4. **Simplicity:** Compile-time decision; zero runtime overhead for stack path
5. **Maintainability:** No platform-specific code; single implementation

This is the **long-term solution** that avoids technical debt while maintaining HyperStream's performance-first design philosophy.

