#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)


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

// Raw SSE2 entry points operating on 64-bit words. Unaligned loads/stores are used (loadu/storeu).
// For GCC/Clang these are defined with function-level target attributes; on MSVC they are defined in .cpp TUs.
void BindWords(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t word_count);
std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b, std::size_t word_count);

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("sse2"))) inline void BindWords(const std::uint64_t* a, const std::uint64_t* b,
                                                       std::uint64_t* out, std::size_t word_count) {
  const std::size_t sse2_words = (word_count / 2) * 2;
  std::size_t i = 0;
  for (; i < sse2_words; i += 2) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
    __m128i vout = _mm_xor_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), vout);
  }
  for (; i < word_count; ++i) out[i] = a[i] ^ b[i];
}

__attribute__((target("sse2"))) inline std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b,
                                                                 std::size_t word_count) {
  const std::size_t sse2_words = (word_count / 2) * 2;
  std::size_t total = 0; std::size_t i = 0;
  for (; i < sse2_words; i += 2) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
    __m128i vx = _mm_xor_si128(va, vb);
    std::uint64_t words[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(words), vx);
    total += __builtin_popcountll(words[0]);
    total += __builtin_popcountll(words[1]);
  }
  for (; i < word_count; ++i) total += __builtin_popcountll(a[i] ^ b[i]);
  return total;
}
#endif

// SSE2 implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
void BindSSE2(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
              core::HyperVector<Dim, bool>* out) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  auto& out_words = out->Words();
  BindWords(a_words.data(), b_words.data(), out_words.data(), a_words.size());
}

// SSE2 implementation of Hamming distance.
// Uses hardware popcount when available, scalar fallback otherwise.
template <std::size_t Dim>
std::size_t HammingDistanceSSE2(const core::HyperVector<Dim, bool>& a,
                                const core::HyperVector<Dim, bool>& b) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  return HammingWords(a_words.data(), b_words.data(), a_words.size());
}

}  // namespace sse2



}  // namespace backend
}  // namespace hyperstream

#endif // x86/x64 guard

