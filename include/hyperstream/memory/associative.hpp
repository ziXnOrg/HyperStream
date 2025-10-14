#pragma once

// =============================================================================
// File:        include/hyperstream/memory/associative.hpp
// Overview:    Fixed-capacity prototype, cluster, and cleanup associative
//              memories over binary HyperVectors.
// Mathematical Foundation: Nearest-neighbour by Hamming; majority thresholding
//              for clusters; direct match restore for cleanup memory.
// Security Considerations: Fixed-capacity containers; no exceptions on hot
//              paths; inputs validated for raw load; noexcept where safe.
// Performance Considerations: Packed 64-bit words for HV; arrays for locality;
//              saturating counters optional via compile-time flag.
//              Hybrid allocation: stack for small sizes, heap for large.
// Examples:    See io/serialization.hpp for persistence helpers.
// =============================================================================
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <vector>

#include "hyperstream/config.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

namespace hyperstream::memory {

// =============================================================================
// Hybrid Allocation Strategy
// =============================================================================
// Windows default stack size: 1 MB (1,048,576 bytes)
// Linux/macOS default: 8 MB
// Conservative threshold: 512 KB (524,288 bytes) to leave headroom for:
// - Other stack variables
// - Nested function calls
// - Compiler padding/alignment
//
// For sizes ≤ 512 KB: use std::array (stack allocation, zero overhead)
// For sizes > 512 KB: use std::vector (heap allocation, one-time cost)
//
// This ensures:
// - Common small/medium dimensions remain stack-allocated (optimal performance)
// - Large dimensions use heap (cross-platform compatibility)
// - No Windows-specific workarounds needed
// =============================================================================

namespace detail {

// Threshold for hybrid allocation: 512 KB
inline constexpr std::size_t kHybridAllocThresholdBytes = 512U * 1024U;

// Helper: choose std::array or std::vector based on compile-time size
template <typename T, std::size_t N>
struct HybridStorage {
  static constexpr std::size_t kTotalBytes = sizeof(T) * N;
  static constexpr bool kUseStack = (kTotalBytes <= kHybridAllocThresholdBytes);

  using type = std::conditional_t<kUseStack, std::array<T, N>, std::vector<T>>;
};

template <typename T, std::size_t N>
using HybridStorageT = typename HybridStorage<T, N>::type;

// Helper: initialize storage (no-op for std::array, reserve for std::vector)
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

/**
 * @brief Fixed-capacity prototype associative memory (nearest neighbour by Hamming).
 *
 * @tparam Dim     Hypervector dimension (bits)
 * @tparam Capacity Maximum number of (prototype,label) entries; no eviction policy.
 *
 * Invariants and behavior:
 * - Capacity is fixed at compile time; Learn() returns false when full (no eviction or
 * replacement).
 * - When size()==0, Classify() returns the provided default_label (no computation performed).
 * - If Capacity==0, all mutating operations fail and size() remains 0 (compile-time constant
 * capacity).
 * - Thread-safety: not thread-safe. External synchronization is required for concurrent access.
 *
 * Complexity (binary HyperVector):
 * - Learn: O(1) append
 * - Classify: O(size * Dim/64) Hamming distance over packed uint64_t words
 *
 * Memory allocation strategy:
 * - Uses hybrid allocation (stack for small sizes ≤ 512 KB, heap for large)
 * - Common dimensions (e.g., 10000×256) remain stack-allocated for optimal performance
 * - Large dimensions (e.g., 10000×1024) use heap to avoid Windows stack overflow
 */
template <std::size_t Dim, std::size_t Capacity>
class PrototypeMemory {
 public:
  struct Entry {
    std::uint64_t label = 0;
    core::HyperVector<Dim, bool> hv;
  };

  PrototypeMemory() : entries_(detail::MakeHybridStorage<Entry, Capacity>()) {}

  bool Learn(std::uint64_t label, const core::HyperVector<Dim, bool>& hypervector) {
    if (size_ >= Capacity) {
      return false;
    }
    entries_[size_].label = label;
    entries_[size_].hv = hypervector;
    ++size_;
    return true;
  }

  [[nodiscard]] std::uint64_t Classify(const core::HyperVector<Dim, bool>& query,
                         std::uint64_t default_label = 0) const {
    if (size_ == 0) {
      return default_label;
    }
    std::size_t best_index = 0;
    std::size_t best_match = 0;
    for (std::size_t i = 0; i < size_; ++i) {
      const std::size_t dist = core::HammingDistance(query, entries_[i].hv);
      const std::size_t match = Dim - dist;
      if (match > best_match) {
        best_match = match;
        best_index = i;
      }
    }
    return entries_[best_index].label;
  }

  // Overload: classify using a caller-provided distance functor.
  // DistFn must be callable as: size_t dist(const HV&, const HV&)
  template <typename DistFn, typename = std::enable_if_t<std::is_invocable_r_v<
                                 std::size_t, DistFn, const core::HyperVector<Dim, bool>&,
                                 const core::HyperVector<Dim, bool>&>>>
  [[nodiscard]] std::uint64_t Classify(const core::HyperVector<Dim, bool>& query, const DistFn& dist_fn,
                         std::uint64_t default_label = 0) const {
    if (size_ == 0) {
      return default_label;
    }
    std::size_t best_index = 0;
    std::size_t best_match = 0;
    for (std::size_t i = 0; i < size_; ++i) {
      const std::size_t dist = dist_fn(query, entries_[i].hv);
      const std::size_t match = Dim - dist;
      if (match > best_match) {
        best_match = match;
        best_index = i;
      }
    }
    return entries_[best_index].label;
  }

  [[nodiscard]] std::size_t Size() const noexcept {
    return size_;
  }
  /**
   * @brief Read-only access to the underlying entries buffer.
   * Returns a pointer to an array of size Capacity; only the first size() entries are valid.
   */
  [[nodiscard]] const Entry* Data() const noexcept {
    return entries_.data();
  }



 private:
  detail::HybridStorageT<Entry, Capacity> entries_;
  std::size_t size_ = 0;
};

/**
 * @brief Fixed-capacity cluster memory with additive counters and thresholding.
 *
 * @tparam Dim      Hypervector dimension (bits)
 * @tparam Capacity Maximum number of clusters; no eviction policy.
 *
 * Invariants and behavior:
 * - Capacity is fixed at compile time; Update() returns false when full.
 * - When size()==0, Finalize() produces an all-zero vector; no-op for unknown label.
 * - If Capacity==0, all mutating operations fail and size() remains 0.
 * - Thread-safety: not thread-safe. External synchronization is required.
 *
 * Complexity:
 * - Update:   O(Dim) to adjust counters per bit
 * - Finalize: O(Dim) to threshold counters into a binary HyperVector
 *
 * Memory allocation strategy:
 * - Uses hybrid allocation (stack for small sizes ≤ 512 KB, heap for large)
 * - The sums_ array is the largest: Capacity × Dim × sizeof(int) bytes
 * - Example: ClusterMemory<10000, 16> ≈ 625 KB → heap; <65536, 16> ≈ 4.2 MB → heap
 */
template <std::size_t Dim, std::size_t Capacity>
class ClusterMemory {
 public:
  ClusterMemory()
      : labels_(detail::MakeHybridStorage<std::uint64_t, Capacity>()),
        counts_(detail::MakeHybridStorage<int, Capacity>()),
        sums_(detail::MakeHybridStorage<int, Capacity * Dim>()) {}

  bool Update(std::uint64_t label, const core::HyperVector<Dim, bool>& hypervector) {
    int index = FindIndex(label);
    if (index < 0) {
      if (size_ >= Capacity) {
        return false;
      }
      index = static_cast<int>(size_);
      labels_[index] = label;
      counts_[index] = 0;
      // sums_ already zero-initialized
      ++size_;
    }

    for (std::size_t bit = 0; bit < Dim; ++bit) {
      sums_[index * Dim + bit] += hypervector.GetBit(bit) ? 1 : -1;
    }
    ++counts_[index];
    return true;
  }

  void ApplyDecay(float decay_factor) {
    if (decay_factor < 0.0f || decay_factor > 1.0f) {
      return;
    }
    for (std::size_t i = 0; i < size_; ++i) {
      for (std::size_t bit = 0; bit < Dim; ++bit) {
        const std::size_t idx = i * Dim + bit;
        sums_[idx] = static_cast<int>(static_cast<float>(sums_[idx]) * decay_factor);
      }
      counts_[i] = static_cast<int>(static_cast<float>(counts_[i]) * decay_factor);
    }
  }

  void Finalize(std::uint64_t label, core::HyperVector<Dim, bool>* out) const {
    const int index = FindIndex(label);
    out->Clear();
    if (index < 0) {
      return;
    }
    for (std::size_t bit = 0; bit < Dim; ++bit) {
      out->SetBit(bit, sums_[index * Dim + bit] >= 0);
    }
  }

  /** Lightweight read-only view of internal buffers (for serialization). */
  struct View {
    const std::uint64_t* labels;
    const int* counts;
    const int* sums;  // length Capacity*Dim, row-major per cluster
    std::size_t size;
  };

  /** Returns a read-only view over labels, counts, and sums; first size() clusters valid. */
  [[nodiscard]] View GetView() const noexcept {
    return View{labels_.data(), counts_.data(), sums_.data(), size_};
  }

  // Backward-compatible alias removed to avoid type name collision in MSVC

  /**
   * @brief Load raw internal buffers. Intended for serialization; validates sizes.
   * Precondition: size()==0. Returns false on invalid input.
   */
  struct LoadRawArgs {
    const std::uint64_t* labels;
    const int* label_counts;
    const int* bit_sums;
    std::size_t num_items;
  };

  bool LoadRaw(const LoadRawArgs& args) noexcept {
    if (size_ != 0 || args.labels == nullptr || args.label_counts == nullptr || args.bit_sums == nullptr) {
      return false;
    }
    if (args.num_items > Capacity) return false;
    for (std::size_t i = 0; i < args.num_items; ++i) {
      labels_[i] = args.labels[i];
      counts_[i] = args.label_counts[i];
      // copy Dim counters for cluster i
      for (std::size_t bit = 0; bit < Dim; ++bit) {
        sums_[i * Dim + bit] = args.bit_sums[i * Dim + bit];
      }
    }
    size_ = args.num_items;
    return true;
  }



  [[nodiscard]] std::size_t Size() const noexcept {
    return size_;
  }

 private:
  [[nodiscard]] int FindIndex(std::uint64_t label) const noexcept {
    for (std::size_t i = 0; i < size_; ++i) {
      if (labels_[i] == label) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  detail::HybridStorageT<std::uint64_t, Capacity> labels_;
  detail::HybridStorageT<int, Capacity> counts_;
  detail::HybridStorageT<int, Capacity * Dim> sums_;
  std::size_t size_ = 0;
};

/**
 * @brief Fixed-capacity cleanup memory (dictionary) restoring to nearest stored vector.
 *
 * @tparam Dim      Hypervector dimension (bits)
 * @tparam Capacity Maximum number of stored items; no eviction policy.
 *
 * Invariants and behavior:
 * - Capacity is fixed at compile time; Insert() returns false when full.
 * - When size()==0, Restore() returns the caller-provided fallback.
 * - If Capacity==0, all mutating operations fail and size() remains 0.
 * - Thread-safety: not thread-safe. External synchronization is required.
 *
 * Complexity:
 * - Insert:  O(1)
 * - Restore: O(size * Dim/64) Hamming distance over packed uint64_t words
 *
 * Memory allocation strategy:
 * - Uses hybrid allocation (stack for small sizes ≤ 512 KB, heap for large)
 * - Example: CleanupMemory<10000, 256> ≈ 316 KB → stack; <10000, 1024> ≈ 1.23 MB → heap
 */
template <std::size_t Dim, std::size_t Capacity>
class CleanupMemory {
 public:
  CleanupMemory() : entries_(detail::MakeHybridStorage<core::HyperVector<Dim, bool>, Capacity>()) {}

  bool Insert(const core::HyperVector<Dim, bool>& hypervector) {
    if (size_ >= Capacity) {
      return false;
    }
    entries_[size_] = hypervector;
    ++size_;
    return true;
  }

  struct RestoreArgs {
    const core::HyperVector<Dim, bool>& noisy;
    const core::HyperVector<Dim, bool>& fallback;
  };

  core::HyperVector<Dim, bool> Restore(const RestoreArgs& args) const {
    if (size_ == 0) {
      return args.fallback;
    }
    std::size_t best_index = 0;
    std::size_t best_match = 0;
    for (std::size_t i = 0; i < size_; ++i) {
      const std::size_t dist = core::HammingDistance(args.noisy, entries_[i]);
      const std::size_t match = Dim - dist;
      if (match > best_match) {
        best_match = match;
        best_index = i;
      }


    }
    return entries_[best_index];
  }

  [[nodiscard]] std::size_t Size() const noexcept {
    return size_;
  }



 private:
  detail::HybridStorageT<core::HyperVector<Dim, bool>, Capacity> entries_;
  std::size_t size_ = 0;
};

}  // namespace hyperstream::memory
