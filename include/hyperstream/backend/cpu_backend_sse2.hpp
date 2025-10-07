#pragma once

// SSE2-accelerated backend primitives for x86-64 platforms.
// Provides fallback when AVX2 is unavailable. Uses 128-bit SIMD operations.
// Compatible with all x86-64 CPUs (SSE2 mandatory since AMD64/Intel 64).

#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream {
namespace backend {
namespace sse2 {

// SSE2 implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
void BindSSE2(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
              core::HyperVector<Dim, bool>* out) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  auto& out_words = out->Words();

  const std::size_t num_words = a_words.size();
  const std::size_t sse2_words = (num_words / 2) * 2;  // Process 2 uint64_t at a time

  std::size_t i = 0;
  // SSE2 path: 2 uint64_t (128 bits) per iteration.
  for (; i < sse2_words; i += 2) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a_words[i]));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b_words[i]));
    __m128i vout = _mm_xor_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&out_words[i]), vout);
  }

  // Scalar tail for remaining words.
  for (; i < num_words; ++i) {
    out_words[i] = a_words[i] ^ b_words[i];
  }
}

// SSE2 implementation of Hamming distance.
// Uses hardware popcount when available, scalar fallback otherwise.
template <std::size_t Dim>
std::size_t HammingDistanceSSE2(const core::HyperVector<Dim, bool>& a,
                                const core::HyperVector<Dim, bool>& b) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();

  const std::size_t num_words = a_words.size();
  const std::size_t sse2_words = (num_words / 2) * 2;

  std::size_t total = 0;
  std::size_t i = 0;

  // SSE2 path: XOR 2 uint64_t at a time, then scalar popcount on each.
  for (; i < sse2_words; i += 2) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a_words[i]));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b_words[i]));
    __m128i vxor = _mm_xor_si128(va, vb);

    // Extract two uint64_t and popcount each.
    std::uint64_t words[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(words), vxor);

#if defined(__GNUC__) || defined(__clang__)
    total += __builtin_popcountll(words[0]);
    total += __builtin_popcountll(words[1]);
#elif defined(_MSC_VER)
    total += __popcnt64(words[0]);
    total += __popcnt64(words[1]);
#else
    for (int j = 0; j < 2; ++j) {
      std::uint64_t x = words[j];
      while (x) {
        x &= (x - 1);
        ++total;
      }
    }
#endif
  }

  // Scalar tail.
  for (; i < num_words; ++i) {
    const std::uint64_t xor_word = a_words[i] ^ b_words[i];
#if defined(__GNUC__) || defined(__clang__)
    total += __builtin_popcountll(xor_word);
#elif defined(_MSC_VER)
    total += __popcnt64(xor_word);
#else
    std::uint64_t x = xor_word;
    while (x) {
      x &= (x - 1);
      ++total;
    }
#endif
  }

  return total;
}

}  // namespace sse2
}  // namespace backend
}  // namespace hyperstream
