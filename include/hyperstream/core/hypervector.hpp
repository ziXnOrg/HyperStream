#pragma once

// Core hypervector type.
// Binary specialization packs bits into 64-bit words for compact storage and efficient
// bitwise operations. Complex variant stores elements contiguously. Header-only;
// no dynamic allocation in hot paths.

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <complex>
#include <stdexcept>

namespace hyperstream {
namespace core {

// Primary template (non-binary). Stores exactly Dim elements of type T.
// Header-only and constexpr-friendly.

template <std::size_t Dim, typename T>
class HyperVector {
 public:
  using value_type = T;
  static_assert(Dim > 0, "HyperVector dimension must be > 0");

  constexpr HyperVector() : data_{} {}

  // Size (number of logical elements; equals Dim).
  [[nodiscard]] static constexpr std::size_t Size() { return Dim; }

  // Element accessors.
  [[nodiscard]] inline const T& operator[](std::size_t i) const {
    return data_[i];
  }
  inline T& operator[](std::size_t i) { return data_[i]; }

  // Raw access for advanced uses (SIMD kernels, etc.).
  [[nodiscard]] inline const std::array<T, Dim>& Raw() const { return data_; }
  inline std::array<T, Dim>& Raw() { return data_; }

 private:
  std::array<T, Dim> data_{};  // contiguous storage for general T
};

// Binary specialization (T = bool): bit-packed into 64-bit words.
// Exposes bit-level accessors; algebraic operations live in core ops.

template <std::size_t Dim>
class HyperVector<Dim, bool> {
 public:
  using value_type = bool;
  static_assert(Dim > 0, "HyperVector dimension must be > 0");

  // Number of bits per storage word.
  static constexpr std::size_t kWordBits = 64;
  // Number of words required to store Dim bits.
  static constexpr std::size_t kWordCount = (Dim + kWordBits - 1) / kWordBits;

  constexpr HyperVector() : word_{} {}

  // Dimension (bits).
  [[nodiscard]] static constexpr std::size_t Size() { return Dim; }
  // Number of storage words.
  [[nodiscard]] static constexpr std::size_t WordCount() { return kWordCount; }

  // Clear all bits to 0.
  inline void Clear() {
    for (std::size_t w = 0; w < kWordCount; ++w) {
      word_[w] = 0ULL;
    }
  }

  // Get/Set individual bit by index [0, Dim).
  [[nodiscard]] inline bool GetBit(std::size_t bit_index) const {
    const auto [w, mask] = WordAndMask(bit_index);
    return (word_[w] & mask) != 0ULL;
  }

  inline void SetBit(std::size_t bit_index, bool value) {
    const auto [w, mask] = WordAndMask(bit_index);
    if (value) {
      word_[w] |= mask;
    } else {
      word_[w] &= ~mask;
    }
  }

  // Raw word access for backends/ops.
  [[nodiscard]] inline const std::array<std::uint64_t, kWordCount>& Words() const {
    return word_;
  }
  inline std::array<std::uint64_t, kWordCount>& Words() { return word_; }

 private:
  // Compute word index and bit mask for a bit position.
  [[nodiscard]] static inline std::pair<std::size_t, std::uint64_t> WordAndMask(
      std::size_t bit_index) {
    // Bounds check for debug builds.
    if (bit_index >= Dim) {
      throw std::out_of_range("HyperVector<bool>: bit index out of range");
    }
    const std::size_t w = bit_index / kWordBits;
    const std::size_t b = bit_index % kWordBits;
    const std::uint64_t mask = (1ULL << b);
    return {w, mask};
  }

  alignas(64) std::array<std::uint64_t, kWordCount> word_{};  // bit-packed storage (64B aligned)
};

// Common typedefs for convenience (these are suggestions; users can instantiate directly).
using Binary10k = HyperVector<10000, bool>;
using Complex5k = HyperVector<5000, std::complex<float>>;

}  // namespace core
}  // namespace hyperstream
