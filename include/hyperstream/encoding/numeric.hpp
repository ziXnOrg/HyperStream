#pragma once

// Numeric encoders: thermometer (scalar) and random projection (vector).
// Header-only; deterministic with explicit seeds; no dynamic allocation.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/item_memory.hpp"

namespace hyperstream {
namespace encoding {

namespace detail_numeric {

// Low-discrepancy order (Van der Corput-inspired) used for thermometer mapping.
template <std::size_t Dim>
inline std::array<std::size_t, Dim> BuildOrder() {
  std::array<std::size_t, Dim> order{};
  for (std::size_t i = 0; i < Dim; ++i) {
    std::size_t value = 0;
    std::size_t bits = i;
    for (std::size_t b = 0; (1ULL << b) <= Dim * 2; ++b) {
      value = (value << 1) | (bits & 1ULL);
      bits >>= 1U;
    }
    order[i] = value % Dim;
  }
  std::array<bool, Dim> used{};
  for (std::size_t i = 0; i < Dim; ++i) {
    std::size_t idx = order[i];
    if (idx >= Dim || used[idx]) {
      idx = 0;
      while (used[idx]) idx++;
      order[i] = idx;
    }
    used[order[i]] = true;
  }
  return order;
}

}  // namespace detail_numeric

/**
 * @brief Thermometer encoder for scalar values.
 * @tparam Dim Hypervector dimension
 *
 * Maps x in [min,max] to k= floor(((x-min)/(max-min)) * Dim) ones distributed by
 * a low-discrepancy order. Values outside range clamp to 0 or Dim.
 *
 * Thread-safety: stateless after construction; reentrant.
 * Complexity: O(Dim).
 */
template <std::size_t Dim>
class ThermometerEncoder {
 public:
  ThermometerEncoder(double min, double max)
      : min_(min), max_(max), order_(detail_numeric::BuildOrder<Dim>()) {}

  void Encode(double x, core::HyperVector<Dim, bool>* out) const {
    out->Clear();
    if (!(max_ > min_)) {
      return;  // degenerate range => zero vector
    }
    double p = (x - min_) / (max_ - min_);
    if (p < 0.0) p = 0.0;
    if (p > 1.0) p = 1.0;
    const std::size_t k = static_cast<std::size_t>(p * static_cast<double>(Dim));
    for (std::size_t i = 0; i < k && i < Dim; ++i) {
      out->SetBit(order_[i], true);
    }
  }

 private:
  double min_;
  double max_;
  std::array<std::size_t, Dim> order_;
};

/**
 * @brief Random projection encoder for dense float vectors.
 *
 * For each input index i, derives a deterministic basis hypervector H_i via ItemMemory
 * and accumulates signed contributions into per-bit counters. Thresholding (>0) yields the
 * output binary HyperVector.
 *
 * Thread-safety: stateless after construction; reentrant. No heap allocation.
 * Complexity: O(n * Dim) naive per-bit accumulation.
 */
template <std::size_t Dim>
class RandomProjectionEncoder {
 public:
  explicit RandomProjectionEncoder(std::uint64_t seed)
      : im_(seed ^ 0xa5a5a5a5a5a5a5a5ULL) {}

  void Encode(const float* data, std::size_t n, core::HyperVector<Dim, bool>* out) const {
    // Per-bit floating counters avoid arbitrary integer scaling.
    std::array<float, Dim> acc{};  // zero-initialized

    for (std::size_t i = 0; i < n; ++i) {
      const float v = data[i];
      if (v == 0.0f) continue;
      core::HyperVector<Dim, bool> basis;
      im_.EncodeId(static_cast<std::uint64_t>(i), &basis);
      for (std::size_t bit = 0; bit < Dim; ++bit) {
        acc[bit] += basis.GetBit(bit) ? v : -v;
      }
    }

    out->Clear();
    for (std::size_t bit = 0; bit < Dim; ++bit) {
      // Strict > 0 ensures empty inputs lead to all-zero output.
      out->SetBit(bit, acc[bit] > 0.0f);
    }
  }

 private:
  ItemMemory<Dim> im_;
};

}  // namespace encoding
}  // namespace hyperstream

