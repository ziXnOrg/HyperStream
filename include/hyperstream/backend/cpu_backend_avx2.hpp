#pragma once

// =============================================================================
// File:        include/hyperstream/backend/cpu_backend_avx2.hpp
// Overview:    AVX2-accelerated primitives (Bind, Hamming) for x86-64.
// Mathematical Foundation: XOR-bind, Hamming via Harley-Seal popcount scheme.
// Security Considerations: Unaligned loads/stores; contiguous HyperVector words;
//              noexcept functions; compile-time target("avx2") where supported.
// Performance Considerations: 256-bit vector ops; loop unrolling per 256-bit
//              lanes; scalar tail handling.
// Examples:    See backend/cpu_backend.hpp for dispatch usage.
// =============================================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

// AVX2-accelerated backend primitives for x86-64 platforms.
// Implements Bind (XOR) and Hamming distance using 256-bit SIMD operations.
// Harley-Seal algorithm for efficient popcount. Requires AVX2 CPU support.
//
// Invariants and I/O contract for HyperStream SIMD backends:
// - Unaligned memory semantics: all vector loads/stores use loadu/storeu; callers need not ensure
//   32-byte alignment. Implementations may opportunistically use aligned ops internally when safe.
// - Contiguous word layout: HyperVector<Dim,bool>::Words() exposes a contiguous array of uint64_t
//   words representing the bit-packed hypervector. Functions here operate exclusively on that
//   layout.
// - Safe tail handling: Dimensions that are not multiples of 64 bits are handled by masking the
//   final word in the HyperVector representation; SIMD loops process full words and scalar-tail
//   code completes the remainder.
// - Compiler targets: On GCC/Clang we use function-level target attributes ("avx2"). On MSVC, the
//   raw ISA entry points are provided from .cpp translation units compiled with /arch:AVX2.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <array>
#include <bit>
#include <span>
#include <immintrin.h>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream::backend::avx2 {

// Portable 64-bit popcount with C++20 std::popcount when available; falls back otherwise.
inline std::size_t Popcount64(std::uint64_t value) noexcept {
#if defined(__cpp_lib_bitops) && (__cpp_lib_bitops >= 201907L)
  return static_cast<std::size_t>(std::popcount(value));
#elif defined(_MSC_VER)
  return static_cast<std::size_t>(__popcnt64(value));
#else
  return static_cast<std::size_t>(__builtin_popcountll(value));
#endif
}

/// @brief XOR-bind two arrays of 64-bit words using AVX2 with unaligned IO.
/// Span-based overload (preferred in headers) for bounds-aware access.
/// @note Uses _mm256_set_epi64x + lane extraction; no alignment preconditions.
void BindWords(std::span<const std::uint64_t> lhs_words,
               std::span<const std::uint64_t> rhs_words,
               std::span<std::uint64_t> out) noexcept;

// Pointer-based overload retained for MSVC TU linkage (src/backend/bind_avx2.cpp)
void BindWords(const std::uint64_t* lhs_words, const std::uint64_t* rhs_words, std::uint64_t* out,
               std::size_t word_count) noexcept;

/// @brief Compute Hamming distance between two word arrays using AVX2 with unaligned IO.
/// Span-based overload (preferred in headers).
[[nodiscard]] std::size_t HammingWords(std::span<const std::uint64_t> lhs_words,
                                       std::span<const std::uint64_t> rhs_words) noexcept;

// Pointer-based overload retained for MSVC TU linkage
[[nodiscard]] std::size_t HammingWords(const std::uint64_t* lhs_words, const std::uint64_t* rhs_words,
                                       std::size_t word_count) noexcept;

// Harley-Seal popcount for __m256i (256-bit vector).
// Uses CSA (Carry-Save Adder) approach to count bits in parallel.
#if defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER))
__attribute__((target("avx2"))) inline std::uint64_t Popcount256(__m256i vector_value) noexcept {
  // Step 1: Count bits in each byte using SSSE3 lookup table approach.
  const __m256i lookup = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1,
                                          2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
  static constexpr int kLowNibbleMask = 0x0F;
  const __m256i low_mask = _mm256_set1_epi8(kLowNibbleMask);

  __m256i low_part = _mm256_and_si256(vector_value, low_mask);
  __m256i high_part = _mm256_and_si256(_mm256_srli_epi16(vector_value, 4), low_mask);
  __m256i popcnt_lo = _mm256_shuffle_epi8(lookup, low_part);
  __m256i popcnt_hi = _mm256_shuffle_epi8(lookup, high_part);
  __m256i sum = _mm256_add_epi8(popcnt_lo, popcnt_hi);

  // Step 2: Horizontal sum across all bytes.
  // Use SAD (Sum of Absolute Differences) to reduce 8-bit counts to 64-bit.
  __m256i sad = _mm256_sad_epu8(sum, _mm256_setzero_si256());

  // Extract four 64-bit counts and sum them.
  const __m128i sad_lo = _mm256_extracti128_si256(sad, 0);
  const __m128i sad_hi = _mm256_extracti128_si256(sad, 1);
  const auto lo0 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(sad_lo));
  const auto hi0 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(sad_lo, 8)));
  const auto lo1 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(sad_hi));
  const auto hi1 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(sad_hi, 8)));
  return lo0 + hi0 + lo1 + hi1;
}
#endif

#if defined(__GNUC__) || (defined(__clang__) && !defined(_MSC_VER))
__attribute__((target("avx2"))) inline void BindWords(std::span<const std::uint64_t> lhs_words,
                                                      std::span<const std::uint64_t> rhs_words,
                                                      std::span<std::uint64_t> out) noexcept {
  static constexpr std::size_t kWordsPer256Bit = 4U;
  const std::size_t word_count = lhs_words.size();
  const std::size_t vector_loop_words = (word_count / kWordsPer256Bit) * kWordsPer256Bit;
  std::size_t word_index = 0;
  for (; word_index < vector_loop_words; word_index += kWordsPer256Bit) {
    __m256i vec_lhs = _mm256_set_epi64x(
      static_cast<std::int64_t>(lhs_words[word_index + 3]),
      static_cast<std::int64_t>(lhs_words[word_index + 2]),
      static_cast<std::int64_t>(lhs_words[word_index + 1]),
      static_cast<std::int64_t>(lhs_words[word_index + 0]));
    __m256i vec_rhs = _mm256_set_epi64x(
      static_cast<std::int64_t>(rhs_words[word_index + 3]),
      static_cast<std::int64_t>(rhs_words[word_index + 2]),
      static_cast<std::int64_t>(rhs_words[word_index + 1]),
      static_cast<std::int64_t>(rhs_words[word_index + 0]));
    __m256i vec_xor = _mm256_xor_si256(vec_lhs, vec_rhs);
    const __m128i x_lo = _mm256_extracti128_si256(vec_xor, 0);
    const __m128i x_hi = _mm256_extracti128_si256(vec_xor, 1);
    const auto word0 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(x_lo));
    const auto word1 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(x_lo, 8)));
    const auto word2 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(x_hi));
    const auto word3 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(x_hi, 8)));
    out[word_index + 0] = word0;
    out[word_index + 1] = word1;
    out[word_index + 2] = word2;
    out[word_index + 3] = word3;
  }
  for (; word_index < word_count; ++word_index) {
    out[word_index] = lhs_words[word_index] ^ rhs_words[word_index];
  }
}

__attribute__((target("avx2"))) inline std::size_t HammingWords(std::span<const std::uint64_t> lhs_words,
                                                                std::span<const std::uint64_t> rhs_words) noexcept {
  static constexpr std::size_t kWordsPer256Bit = 4U;
  const std::size_t word_count = lhs_words.size();
  const std::size_t vector_loop_words = (word_count / kWordsPer256Bit) * kWordsPer256Bit;
  std::size_t total = 0;
  std::size_t word_index = 0;
  for (; word_index < vector_loop_words; word_index += kWordsPer256Bit) {
    __m256i vec_lhs = _mm256_set_epi64x(
      static_cast<std::int64_t>(lhs_words[word_index + 3]),
      static_cast<std::int64_t>(lhs_words[word_index + 2]),
      static_cast<std::int64_t>(lhs_words[word_index + 1]),
      static_cast<std::int64_t>(lhs_words[word_index + 0]));
    __m256i vec_rhs = _mm256_set_epi64x(
      static_cast<std::int64_t>(rhs_words[word_index + 3]),
      static_cast<std::int64_t>(rhs_words[word_index + 2]),
      static_cast<std::int64_t>(rhs_words[word_index + 1]),
      static_cast<std::int64_t>(rhs_words[word_index + 0]));
    __m256i vec_xor = _mm256_xor_si256(vec_lhs, vec_rhs);
    total += Popcount256(vec_xor);
  }
  for (; word_index < word_count; ++word_index) {
    const std::uint64_t xor_word = lhs_words[word_index] ^ rhs_words[word_index];
    total += Popcount64(xor_word);
  }
  return total;
}
#else
// MSVC path: provide span overloads that forward to pointer-based TU symbols
inline void BindWords(std::span<const std::uint64_t> lhs_words,
                     std::span<const std::uint64_t> rhs_words,
                     std::span<std::uint64_t> out) noexcept {
  BindWords(lhs_words.data(), rhs_words.data(), out.data(), lhs_words.size());
}
inline std::size_t HammingWords(std::span<const std::uint64_t> lhs_words,
                                std::span<const std::uint64_t> rhs_words) noexcept {
  return HammingWords(lhs_words.data(), rhs_words.data(), lhs_words.size());
}
#endif

// AVX2 implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
void BindAVX2(const core::HyperVector<Dim, bool>& lhs, const core::HyperVector<Dim, bool>& rhs,
              core::HyperVector<Dim, bool>* out) noexcept {
  const auto& lhs_words = lhs.Words();
  const auto& rhs_words = rhs.Words();
  auto& out_words = out->Words();
  BindWords(std::span<const std::uint64_t>(lhs_words),
            std::span<const std::uint64_t>(rhs_words),
            std::span<std::uint64_t>(out_words));
}

// AVX2 implementation of Hamming distance.
template <std::size_t Dim>
[[nodiscard]] std::size_t HammingDistanceAVX2(const core::HyperVector<Dim, bool>& lhs,
                                              const core::HyperVector<Dim, bool>& rhs) noexcept {
  const auto& lhs_words = lhs.Words();
  const auto& rhs_words = rhs.Words();
  return HammingWords(std::span<const std::uint64_t>(lhs_words),
                      std::span<const std::uint64_t>(rhs_words));
}

}  // namespace hyperstream::backend::avx2

#endif  // x86/x64 guard
