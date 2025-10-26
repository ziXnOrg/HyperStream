#pragma once

// =============================================================================
// File:        include/hyperstream/core/hypervector.hpp
// Overview:    Core HyperVector type. Primary template stores Dim elements of T;
//              bool-specialization packs bits into 64-bit words for efficiency.
// Mathematical Foundation: Bit-packed binary HVs support XOR bind, majority
//              bundling, permutations (rotations), and Hamming/cosine measures.
// Security Considerations: No dynamic allocation in hot paths; bounds checks in
//              debug via exceptions (or terminate with exceptions disabled).
// Performance Considerations: Cache-friendly contiguous storage; 64-bit word
//              packing minimizes memory bandwidth; interfaces are noexcept where
//              safe and constexpr when possible.
// Examples:    See core/ops.hpp for binding, bundling, permutation, similarity.
// =============================================================================

// Core hypervector type.
// Binary specialization packs bits into 64-bit words for compact storage and efficient
// bitwise operations. Complex variant stores elements contiguously. Header-only;
// no dynamic allocation in hot paths.

#include <array>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace hyperstream::core {

// Primary template (non-binary). Stores exactly Dim elements of type T.
// Header-only and constexpr-friendly.

template <std::size_t Dim, typename T>
class HyperVector {
 public:
  using value_type = T;
  static_assert(Dim > 0, "HyperVector dimension must be > 0");

  constexpr HyperVector() = default;

  // Size (number of logical elements; equals Dim).
  [[nodiscard]] static constexpr std::size_t Size() noexcept {
    return Dim;
  }

  // Element accessors.
  [[nodiscard]] constexpr const T& operator[](std::size_t element_index) const {
    return data_[element_index];
  }
  constexpr T& operator[](std::size_t element_index) {
    return data_[element_index];
  }

  // Raw access for advanced uses (SIMD kernels, etc.).
  [[nodiscard]] constexpr const std::array<T, Dim>& Raw() const noexcept {
    return data_;
  }
  constexpr std::array<T, Dim>& Raw() noexcept {
    return data_;
  }

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
  // Target cache line alignment for bit-packed storage
  static constexpr std::size_t kCacheLineBytes = 64U;

  constexpr HyperVector() = default;

  // Dimension (bits).
  [[nodiscard]] static constexpr std::size_t Size() {
    return Dim;
  }
  // Number of storage words.
  [[nodiscard]] static constexpr std::size_t WordCount() noexcept {
    return kWordCount;
  }

  // Clear all bits to 0.
  constexpr void Clear() noexcept { word_.fill(0ULL); }

  // Get/Set individual bit by index [0, Dim).
  [[nodiscard]] constexpr bool GetBit(std::size_t bit_index) const {
    const auto info = WordAndMask(bit_index);
    const std::size_t word_index = info.first;
    const std::uint64_t mask = info.second;
    return (word_[word_index] & mask) != 0ULL;
  }

  constexpr void SetBit(std::size_t bit_index, bool value) {
    const auto info = WordAndMask(bit_index);
    const std::size_t word_index = info.first;
    const std::uint64_t mask = info.second;
    if (value) {
      word_[word_index] |= mask;
    } else {
      word_[word_index] &= ~mask;
    }
  }

  // Raw word access for backends/ops.
  [[nodiscard]] constexpr const std::array<std::uint64_t, kWordCount>& Words() const noexcept {
    return word_;
  }
  constexpr std::array<std::uint64_t, kWordCount>& Words() noexcept {
    return word_;
  }

 private:
  // Compute word index and bit mask for a bit position.
  [[nodiscard]] static constexpr std::pair<std::size_t, std::uint64_t> WordAndMask(
      std::size_t bit_index) {
    // Bounds check for debug builds.
    if (bit_index >= Dim) {
#if defined(__cpp_exceptions)
      throw std::out_of_range("HyperVector<bool>: bit index out of range");
#else
      std::terminate();
#endif
    }
    const std::size_t word_index = bit_index / kWordBits;
    const std::size_t bit_offset = bit_index % kWordBits;
    const std::uint64_t mask = (1ULL << bit_offset);
    return {word_index, mask};
  }

  alignas(kCacheLineBytes) std::array<std::uint64_t, kWordCount> word_{};  // bit-packed storage
};

// Common typedefs for convenience (these are suggestions; users can instantiate directly).
static constexpr std::size_t kBinary10kDim = 10000U;
static constexpr std::size_t kComplex5kDim = 5000U;
using Binary10k = HyperVector<kBinary10kDim, bool>;
using Complex5k = HyperVector<kComplex5kDim, std::complex<float>>;

}  // namespace hyperstream::core
