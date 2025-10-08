#pragma once

// Deterministic item memory mapping symbols to binary HyperVectors.
// Header-only; zero external deps; no heap allocation.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream {
namespace encoding {

namespace detail_itemmemory {

constexpr std::uint64_t kGoldenGamma = 0x9e3779b97f4a7c15ULL;

inline std::uint64_t SplitMix64Step(std::uint64_t& state) {
  state += kGoldenGamma;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

inline std::uint64_t MixSymbol(std::uint64_t seed, std::uint64_t symbol) {
  std::uint64_t s = seed + symbol * 0x94d049bb133111ebULL;
  s ^= (symbol << 32) | (symbol >> 32);
  s *= 0xbf58476d1ce4e5b9ULL;
  return s;
}

inline std::uint64_t Fnv1a64(std::string_view token, std::uint64_t seed) {
  std::uint64_t hash = 1469598103934665603ULL ^ seed;
  for (unsigned char c : token) {
    hash ^= static_cast<std::uint64_t>(c);
    hash *= 1099511628211ULL;
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
  explicit ItemMemory(std::uint64_t seed) : seed_(seed) {}

  /** Encodes a 64-bit id into a binary HyperVector. */
  void EncodeId(std::uint64_t id, core::HyperVector<Dim, bool>* out) const noexcept {
    using detail_itemmemory::MixSymbol;
    using detail_itemmemory::SplitMix64Step;
    static constexpr std::size_t kWordBits = core::HyperVector<Dim, bool>::kWordBits;

    out->Clear();
    std::uint64_t state = MixSymbol(seed_, id);
    auto& words = out->Words();
    for (std::size_t i = 0; i < words.size(); ++i) {
      words[i] = SplitMix64Step(state);
    }
    // Mask trailing bits beyond Dim in the last word.
    const std::size_t excess = words.size() * kWordBits - Dim;
    if (excess > 0) {
      const std::uint64_t mask = (excess == kWordBits) ? 0ULL : (~0ULL >> excess);
      words.back() &= mask;
    }
  }

  /** Encodes a token (string) into a binary HyperVector. */
  void EncodeToken(std::string_view token, core::HyperVector<Dim, bool>* out) const {
    using detail_itemmemory::Fnv1a64;
    const std::uint64_t sym = Fnv1a64(token, seed_ ^ 0x5bf03635f0b7a54dULL);
    EncodeId(sym, out);
  }

 private:
  std::uint64_t seed_;
};

}  // namespace encoding
}  // namespace hyperstream

