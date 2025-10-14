#pragma once

// Deterministic item memory mapping symbols to binary HyperVectors.
// Header-only; zero external deps; no heap allocation.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream::encoding {

namespace detail_itemmemory {

constexpr std::uint64_t kGoldenGamma = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kSplitMixMul1 = 0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t kSplitMixMul2 = 0x94d049bb133111ebULL;
constexpr std::uint32_t kSplitMixShift1 = 30U;
constexpr std::uint32_t kSplitMixShift2 = 27U;
constexpr std::uint32_t kSplitMixShift3 = 31U;
constexpr std::uint32_t kHalfWordBits = 32U;
constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime64 = 1099511628211ULL;
constexpr std::uint64_t kTokenSalt = 0x5bf03635f0b7a54dULL;

[[nodiscard]] inline std::uint64_t SplitMix64Step(std::uint64_t& state) noexcept {
  state += kGoldenGamma;
  std::uint64_t mixed_value = state;
  mixed_value = (mixed_value ^ (mixed_value >> kSplitMixShift1)) * kSplitMixMul1;
  mixed_value = (mixed_value ^ (mixed_value >> kSplitMixShift2)) * kSplitMixMul2;
  return mixed_value ^ (mixed_value >> kSplitMixShift3);
}

[[nodiscard]] inline std::uint64_t MixSymbol(std::uint64_t seed_value, std::uint64_t symbol) noexcept {
  std::uint64_t mixed_value = (seed_value + (symbol * kSplitMixMul2));
  mixed_value ^= (symbol << kHalfWordBits) | (symbol >> kHalfWordBits);
  mixed_value *= kSplitMixMul1;
  return mixed_value;
}

[[nodiscard]] inline std::uint64_t Fnv1a64(std::string_view token, std::uint64_t seed_value) noexcept {
  std::uint64_t hash = kFnvOffsetBasis64 ^ seed_value;
  for (unsigned char char_value : token) {
    hash ^= static_cast<std::uint64_t>(char_value);
    hash *= kFnvPrime64;
  }
  return hash;
}

}  // namespace detail_itemmemory

/**
 * @brief Deterministic item memory mapping ids/tokens to binary HyperVectors.
 *
 * @tparam Dim Hypervector dimension (bits)
 *
 * Properties and behavior:
 * - Fully deterministic for a given seed and input symbol.
 * - No dynamic allocation; writes into caller-provided output vector.
 * - Thread-safety: thread-safe for concurrent reads; no shared mutable state.
 *
 * Complexity (binary HyperVector):
 * - Encode: O(Dim/64) word generation via SplitMix64.
 */
template <std::size_t Dim>
class ItemMemory {
 public:
  explicit ItemMemory(std::uint64_t seed_value) : seed_(seed_value) {}

  /** Encodes a 64-bit identifier into a binary HyperVector. */
  void EncodeId(std::uint64_t identifier, core::HyperVector<Dim, bool>* out) const noexcept {
    using detail_itemmemory::MixSymbol;
    using detail_itemmemory::SplitMix64Step;
    static constexpr std::size_t kWordBits = core::HyperVector<Dim, bool>::kWordBits;

    out->Clear();
    std::uint64_t state = MixSymbol(seed_, identifier);
    auto& words = out->Words();
    for (std::size_t word_index = 0; word_index < words.size(); ++word_index) {
      words[word_index] = SplitMix64Step(state);
    }
    // Mask trailing bits beyond Dim in the last word.
    const std::size_t excess_bits = words.size() * kWordBits - Dim;
    if (excess_bits > 0) {
      const std::uint64_t tail_mask = (~0ULL) >> excess_bits;
      words.back() &= tail_mask;
    }
  }

  /** Encodes a token (string) into a binary HyperVector. */
  void EncodeToken(std::string_view token, core::HyperVector<Dim, bool>* out) const noexcept {
    using detail_itemmemory::Fnv1a64;
    const std::uint64_t sym = Fnv1a64(token, seed_ ^ detail_itemmemory::kTokenSalt);
    EncodeId(sym, out);
  }

 private:
  std::uint64_t seed_;
};

}  // namespace hyperstream::encoding
