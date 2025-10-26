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

namespace hyperstream::encoding {

namespace detail_numeric {

// Low-discrepancy order (Van der Corput-inspired) used for thermometer mapping.
template <std::size_t Dim>
[[nodiscard]] inline std::array<std::size_t, Dim> BuildOrder() noexcept {
  std::array<std::size_t, Dim> order{};
  for (std::size_t index = 0; index < Dim; ++index) {
    std::size_t reversed_bits_value = 0;
    std::size_t remaining_bits = index;
    static constexpr std::uint64_t kGrowthFactor = 2ULL;
    for (std::size_t bit_index = 0; (1ULL << bit_index) <= Dim * kGrowthFactor; ++bit_index) {
      reversed_bits_value = (reversed_bits_value << 1U) | (remaining_bits & 1ULL);
      remaining_bits >>= 1U;
    }
    order[index] = reversed_bits_value % Dim;
  }
  std::array<bool, Dim> used{};
  for (std::size_t index = 0; index < Dim; ++index) {
    std::size_t mapped_index = order[index];
    if (mapped_index >= Dim || used[mapped_index]) {
      mapped_index = 0;
      while (used[mapped_index]) {
        mapped_index++;
      }
      order[index] = mapped_index;
    }
    used[order[index]] = true;
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
  ThermometerEncoder(double min_value, double max_value)
      : min_(min_value), max_(max_value), order_(detail_numeric::BuildOrder<Dim>()) {}

  void Encode(double value, core::HyperVector<Dim, bool>* out) const noexcept {
    out->Clear();
    if (!(max_ > min_)) {
      return;  // degenerate range => zero vector
    }
    double proportion = (value - min_) / (max_ - min_);
    if (proportion < 0.0) {
      proportion = 0.0;
    }
    if (proportion > 1.0) {
      proportion = 1.0;
    }
    const std::size_t num_active_bits = static_cast<std::size_t>(proportion * static_cast<double>(Dim));
    for (std::size_t bit_index = 0; bit_index < num_active_bits && bit_index < Dim; ++bit_index) {
      out->SetBit(order_[bit_index], true);
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
  explicit RandomProjectionEncoder(std::uint64_t seed) : im_(seed ^ 0xa5a5a5a5a5a5a5a5ULL) {}

  void Encode(const float* data_values, std::size_t num_values, core::HyperVector<Dim, bool>* out) const noexcept {
    // Per-bit floating counters avoid arbitrary integer scaling.
    std::array<float, Dim> acc{};  // zero-initialized

    for (std::size_t value_index = 0; value_index < num_values; ++value_index) {
      const float value = data_values[value_index];
      if (value == 0.0f) {
        continue;
      }
      core::HyperVector<Dim, bool> basis;
      im_.EncodeId(static_cast<std::uint64_t>(value_index), &basis);
      for (std::size_t bit_index = 0; bit_index < Dim; ++bit_index) {
        acc[bit_index] += basis.GetBit(bit_index) ? value : -value;
      }
    }

    out->Clear();
    for (std::size_t bit_index = 0; bit_index < Dim; ++bit_index) {
      // Strict > 0 ensures empty inputs lead to all-zero output.
      out->SetBit(bit_index, acc[bit_index] > 0.0f);
    }
  }

 private:
  ItemMemory<Dim> im_;
};

}  // namespace hyperstream::encoding
