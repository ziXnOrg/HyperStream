#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)


// SSE2-accelerated backend primitives for x86-64 platforms.
// Provides fallback when AVX2 is unavailable. Uses 128-bit SIMD operations.
// Compatible with all x86-64 CPUs (SSE2 mandatory since AMD64/Intel 64).
//
// Invariants and I/O contract for HyperStream SIMD backends:
// - Unaligned memory semantics: all vector loads/stores use loadu/storeu; callers need not ensure
//   16-byte alignment. Implementations may opportunistically use aligned ops internally when safe.
// - Contiguous word layout: HyperVector<Dim,bool>::Words() exposes a contiguous array of uint64_t
//   words representing the bit-packed hypervector. Functions here operate exclusively on that layout.
// - Safe tail handling: Dimensions that are not multiples of 64 bits are handled by masking the
//   final word in the HyperVector representation; SIMD loops process full words and scalar-tail code
//   completes the remainder.
// - Compiler targets: On GCC/Clang we use function-level target attributes ("sse2"). On MSVC, the
//   raw ISA entry points are provided from .cpp translation units compiled with /arch:SSE2.

#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream {
namespace backend {
namespace sse2 {

/// @brief XOR-bind two arrays of 64-bit words using SSE2 with unaligned IO.
/// @param a Pointer to first input array (size: word_count)
/// @param b Pointer to second input array (size: word_count)
/// @param out Pointer to output array (size: word_count)
/// @param word_count Number of 64-bit words to process
/// @note Uses _mm_loadu_si128/_mm_storeu_si128; no alignment preconditions.
void BindWords(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t word_count);

/// @brief Compute Hamming distance between two word arrays using SSE2 with unaligned IO.
/// @param a Pointer to first input array (size: word_count)
/// @param b Pointer to second input array (size: word_count)
/// @param word_count Number of 64-bit words to process
/// @return Total number of differing bits across all words
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

