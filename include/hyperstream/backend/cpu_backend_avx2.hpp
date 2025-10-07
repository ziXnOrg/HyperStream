#pragma once

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)


// AVX2-accelerated backend primitives for x86-64 platforms.
// Implements Bind (XOR) and Hamming distance using 256-bit SIMD operations.
// Harley-Seal algorithm for efficient popcount. Requires AVX2 CPU support.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream {
namespace backend {
namespace avx2 {

// Raw AVX2 entry points operating on 64-bit words. Unaligned load/store used.
// For GCC/Clang these are defined with function-level target attributes; on MSVC they are defined in .cpp TUs.
void BindWords(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t word_count);
std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b, std::size_t word_count);

// Harley-Seal popcount for __m256i (256-bit vector).
// Uses CSA (Carry-Save Adder) approach to count bits in parallel.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"))) inline std::uint64_t Popcount256(__m256i v) {
#else
inline std::uint64_t Popcount256(__m256i v) {
#endif
  // Step 1: Count bits in each byte using SSSE3 lookup table approach.
  const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1,
                                          2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  const __m256i low_mask = _mm256_set1_epi8(0x0f);

  __m256i lo = _mm256_and_si256(v, low_mask);
  __m256i hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), low_mask);
  __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, lo);
  __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, hi);
  __m256i sum = _mm256_add_epi8(popcnt_lo, popcnt_hi);

  // Step 2: Horizontal sum across all bytes.
  // Use SAD (Sum of Absolute Differences) to reduce 8-bit counts to 64-bit.
  __m256i sad = _mm256_sad_epu8(sum, _mm256_setzero_si256());

  // Extract four 64-bit counts and sum them.
  std::uint64_t counts[4];
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(counts), sad);
  return counts[0] + counts[1] + counts[2] + counts[3];
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("avx2"))) inline void BindWords(const std::uint64_t* a, const std::uint64_t* b,
                                                       std::uint64_t* out, std::size_t word_count) {
  const std::size_t avx2_words = (word_count / 4) * 4;
  std::size_t i = 0;
  for (; i < avx2_words; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
    __m256i vx = _mm256_xor_si256(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out[i]), vx);
  }
  for (; i < word_count; ++i) out[i] = a[i] ^ b[i];
}

__attribute__((target("avx2"))) inline std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b,
                                                                 std::size_t word_count) {
  const std::size_t avx2_words = (word_count / 4) * 4;
  std::size_t total = 0; std::size_t i = 0;
  for (; i < avx2_words; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
    __m256i vx = _mm256_xor_si256(va, vb);
    total += Popcount256(vx);
  }
  for (; i < word_count; ++i) {
    const std::uint64_t x = a[i] ^ b[i];
    total += __builtin_popcountll(x);
  }
  return total;
}
#endif

// AVX2 implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
void BindAVX2(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
              core::HyperVector<Dim, bool>* out) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  auto& out_words = out->Words();
  BindWords(a_words.data(), b_words.data(), out_words.data(), a_words.size());
}

// AVX2 implementation of Hamming distance.
template <std::size_t Dim>
std::size_t HammingDistanceAVX2(const core::HyperVector<Dim, bool>& a,
                                const core::HyperVector<Dim, bool>& b) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  return HammingWords(a_words.data(), b_words.data(), a_words.size());
}




}  // namespace avx2
}  // namespace backend
}  // namespace hyperstream


#endif // x86/x64 guard
