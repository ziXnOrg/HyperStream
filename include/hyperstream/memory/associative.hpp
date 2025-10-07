#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <memory>

#include "hyperstream/config.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

namespace hyperstream {
namespace memory {

// Prototype associative memory.
// Stores up to Capacity labeled hypervectors and performs nearest-neighbor lookup
// using Hamming similarity. No dynamic allocation; all storage is pre-sized.
template <std::size_t Dim, std::size_t Capacity>
class PrototypeMemory {
 public:
  struct Entry {
    std::uint64_t label = 0;
    core::HyperVector<Dim, bool> hv;
  };

  PrototypeMemory() : entries_(new Entry[Capacity]{}) {}

  bool Learn(std::uint64_t label, const core::HyperVector<Dim, bool>& hv) {
    if (size_ >= Capacity) {
      return false;
    }
    entries_[size_].label = label;
    entries_[size_].hv = hv;
    ++size_;
    return true;
  }

  std::uint64_t Classify(const core::HyperVector<Dim, bool>& query,
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
  template <typename DistFn,
            typename = std::enable_if_t<
                std::is_invocable_r<std::size_t, DistFn,
                                    const core::HyperVector<Dim, bool>&,
                                    const core::HyperVector<Dim, bool>&>::value>>
  std::uint64_t Classify(const core::HyperVector<Dim, bool>& query,
                         DistFn&& dist_fn,
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

  std::size_t size() const {
    return size_;
  }

 private:
  std::unique_ptr<Entry[]> entries_;
  std::size_t size_ = 0;
};

// Cluster memory with decay. Maintains additive counters per bit and label,
// thresholding back to binary on demand. Decay is applied by scaling counters.
template <std::size_t Dim, std::size_t Capacity>
class ClusterMemory {
 public:
  ClusterMemory() : sums_(new int[Capacity * Dim]{}) {}

  bool Update(std::uint64_t label, const core::HyperVector<Dim, bool>& hv) {
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
      sums_[index * Dim + bit] += hv.GetBit(bit) ? 1 : -1;
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

  std::size_t size() const {
    return size_;
  }

 private:
  int FindIndex(std::uint64_t label) const {
    for (std::size_t i = 0; i < size_; ++i) {
      if (labels_[i] == label) {
        return static_cast<int>(i);
      }
    }
    return -1;
  }

  std::array<std::uint64_t, Capacity> labels_{};
  std::array<int, Capacity> counts_{};
  std::unique_ptr<int[]> sums_;
  std::size_t size_ = 0;
};

// Cleanup memory: dictionary of canonical hypervectors. Restores noisy inputs
// by returning the nearest stored vector. Falls back to caller-provided vector
// when dictionary is empty.
template <std::size_t Dim, std::size_t Capacity>
class CleanupMemory {
 public:
  CleanupMemory() = default;

  bool Insert(const core::HyperVector<Dim, bool>& hv) {
    if (size_ >= Capacity) {
      return false;
    }
    entries_[size_] = hv;
    ++size_;
    return true;
  }

  core::HyperVector<Dim, bool> Restore(const core::HyperVector<Dim, bool>& noisy,
                                       const core::HyperVector<Dim, bool>& fallback) const {
    if (size_ == 0) {
      return fallback;
    }
    std::size_t best_index = 0;
    std::size_t best_match = 0;
    for (std::size_t i = 0; i < size_; ++i) {
      const std::size_t dist = core::HammingDistance(noisy, entries_[i]);
      const std::size_t match = Dim - dist;
      if (match > best_match) {
        best_match = match;
        best_index = i;
      }
    }
    return entries_[best_index];
  }

  std::size_t size() const {
    return size_;
  }

 private:
  std::array<core::HyperVector<Dim, bool>, Capacity> entries_{};
  std::size_t size_ = 0;
};

}  // namespace memory
}  // namespace hyperstream
