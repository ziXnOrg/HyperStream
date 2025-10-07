#pragma once

// HyperStream configuration: default dimensions/capacities and profile controls.
// This header provides compile-time constants and lightweight helpers
// without introducing any dynamic behavior or external dependencies.
//
// Profiles (mutually exclusive):
// - HYPERSTREAM_PROFILE_EMBEDDED: conservative defaults for constrained targets
// - HYPERSTREAM_PROFILE_DESKTOP: explicit desktop profile (same as default)
//
// Overrides:
// - HYPERSTREAM_DIM_BITS: override default dimension bits
// - HYPERSTREAM_BACKEND_OVERRIDE: force a backend selection at compile time
//   (values: "scalar", "sse2", "avx2")
// - HYPERSTREAM_FORCE_SCALAR: disable runtime detection and force scalar at runtime

#include <cstddef>
#include <cstdint>

namespace hyperstream {
namespace config {

// ---- Profile selection ----
#if defined(HYPERSTREAM_PROFILE_EMBEDDED)
static constexpr const char kActiveProfile[] = "embedded";
static constexpr bool kForceHeapForLargeStructures = true;
#elif defined(HYPERSTREAM_PROFILE_DESKTOP)
static constexpr const char kActiveProfile[] = "desktop";
static constexpr bool kForceHeapForLargeStructures = false;
#else
static constexpr const char kActiveProfile[] = "desktop"; // default
static constexpr bool kForceHeapForLargeStructures = false;
#endif

// ---- Defaults ----
#if defined(HYPERSTREAM_PROFILE_EMBEDDED)
static constexpr std::size_t kDefaultDimBits = 2048;   // reduced for embedded
static constexpr std::size_t kDefaultCapacity = 16;    // reduced for embedded
#else
#  if defined(HYPERSTREAM_DIM_BITS)
static constexpr std::size_t kDefaultDimBits = static_cast<std::size_t>(HYPERSTREAM_DIM_BITS);
#  else
static constexpr std::size_t kDefaultDimBits = 10000;
#  endif
static constexpr std::size_t kDefaultCapacity = 256;
#endif

// Heap allocation policy threshold: structures >= this many bytes should be heap-allocated
// to avoid stack overflow and improve robustness across toolchains/runtimes.
static constexpr std::size_t kHeapAllocThresholdBytes = 1024; // 1 KiB

// Helpers
constexpr inline bool IsPowerOfTwo(std::size_t x) { return x && ((x & (x - 1)) == 0); }

// Footprint helpers (storage-only estimates)
/// Returns the storage size in bytes of a binary HyperVector with dim_bits.
constexpr inline std::size_t BinaryHyperVectorStorageBytes(std::size_t dim_bits) {
  return ((dim_bits + 63) / 64) * sizeof(std::uint64_t);
}
/// Returns the storage size in bytes of PrototypeMemory<dim_bits,capacity> entries.
constexpr inline std::size_t PrototypeMemoryStorageBytes(std::size_t dim_bits, std::size_t capacity) {
  return capacity * (sizeof(std::uint64_t) + BinaryHyperVectorStorageBytes(dim_bits));
}
/// Returns the storage size in bytes of ClusterMemory<dim_bits,capacity> counters and metadata.
constexpr inline std::size_t ClusterMemoryStorageBytes(std::size_t dim_bits, std::size_t capacity) {
  return capacity * sizeof(std::uint64_t) +  // labels
         capacity * sizeof(int) +            // counts
         capacity * dim_bits * sizeof(int);  // sums
}
/// Returns the storage size in bytes of CleanupMemory<dim_bits,capacity> entries.
constexpr inline std::size_t CleanupMemoryStorageBytes(std::size_t dim_bits, std::size_t capacity) {
  return capacity * BinaryHyperVectorStorageBytes(dim_bits);
}

// Sanity constraints for defaults
static_assert(kDefaultDimBits >= 8, "Default dimension must be >= 8 bits");
static_assert(kDefaultCapacity >= 1, "Default capacity must be >= 1");
static_assert(IsPowerOfTwo(kDefaultCapacity), "Default capacity should be power of two for fast indexing");

} // namespace config
} // namespace hyperstream

