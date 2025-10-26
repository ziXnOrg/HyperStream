#pragma once

// =============================================================================
// File:        include/hyperstream/backend/cpu_backend_sse2.hpp
// Overview:    SSE2-accelerated primitives (Bind, Hamming) for x86-64.
// Mathematical Foundation: XOR-bind, bitwise Hamming via popcount.
// Security Considerations: Unaligned loads/stores; operates on contiguous
//              HyperVector word layout; noexcept functions.
// Performance Considerations: 128-bit vector operations; tail handled scalarly;
//              avoids alignment preconditions via loadu/storeu.
// Examples:    See backend/cpu_backend.hpp for dispatch usage.
// =============================================================================
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

// SSE2-accelerated backend primitives for x86-64 platforms.
// Provides fallback when AVX2 is unavailable. Uses 128-bit SIMD operations.
// Compatible with all x86-64 CPUs (SSE2 mandatory since AMD64/Intel 64).
//
// Invariants and I/O contract for HyperStream SIMD backends:
// - Unaligned memory semantics: all vector loads/stores use loadu/storeu; callers need not ensure
//   16-byte alignment. Implementations may opportunistically use aligned ops internally when safe.
// - Contiguous word layout: HyperVector<Dim,bool>::Words() exposes a contiguous array of uint64_t
//   words representing the bit-packed hypervector. Functions here operate exclusively on that
//   layout.
// - Safe tail handling: Dimensions that are not multiples of 64 bits are handled by masking the
//   final word in the HyperVector representation; SIMD loops process full words and scalar-tail
//   code completes the remainder.
// - Compiler targets: On GCC/Clang we use function-level target attributes ("sse2"). On MSVC, the
//   raw ISA entry points are provided from .cpp translation units compiled with /arch:SSE2.

#include <cstddef>
#include <cstdint>
#include <array>
#include <bit>
#include <span>
#include <emmintrin.h>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream::backend::sse2 {

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

/// @brief XOR-bind two arrays of 64-bit words using SSE2 with unaligned IO.
/// Uses spans for bounds-aware access; callers need only provide contiguous words.
/// @note Uses _mm_loadu_si128/_mm_storeu_si128; no alignment preconditions.
void BindWords(std::span<const std::uint64_t> lhs_words,
               std::span<const std::uint64_t> rhs_words,
               std::span<std::uint64_t> out) noexcept;

// Overload retained for MSVC TU linkage and external callers (src/backend/bind_sse2.cpp)
void BindWords(const std::uint64_t* lhs_words, const std::uint64_t* rhs_words,
               std::uint64_t* out, std::size_t word_count) noexcept;

/// @brief Compute Hamming distance between two word arrays using SSE2 with unaligned IO.
/// Span-based overload (preferred in headers).
[[nodiscard]] std::size_t HammingWords(std::span<const std::uint64_t> lhs_words,
                                       std::span<const std::uint64_t> rhs_words) noexcept;

// Pointer-based overload retained for MSVC TU linkage
[[nodiscard]] std::size_t HammingWords(const std::uint64_t* lhs_words, const std::uint64_t* rhs_words,
                                       std::size_t word_count) noexcept;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((target("sse2"))) inline void BindWords(std::span<const std::uint64_t> lhs_words,
                                                      std::span<const std::uint64_t> rhs_words,
                                                      std::span<std::uint64_t> out) noexcept {
  static constexpr std::size_t kWordsPer128Bit = 2U;
  const std::size_t word_count = lhs_words.size();
  const std::size_t vector_loop_words = (word_count / kWordsPer128Bit) * kWordsPer128Bit;
  std::size_t word_index = 0;
  for (; word_index < vector_loop_words; word_index += kWordsPer128Bit) {
    __m128i vec_lhs = _mm_set_epi64x(static_cast<std::int64_t>(lhs_words[word_index + 1]),
                                     static_cast<std::int64_t>(lhs_words[word_index]));
    __m128i vec_rhs = _mm_set_epi64x(static_cast<std::int64_t>(rhs_words[word_index + 1]),
                                     static_cast<std::int64_t>(rhs_words[word_index]));
    __m128i vec_xor = _mm_xor_si128(vec_lhs, vec_rhs);
    const auto low_word = static_cast<std::uint64_t>(_mm_cvtsi128_si64(vec_xor));
    const __m128i hi_shift = _mm_srli_si128(vec_xor, 8);
    const auto high_word = static_cast<std::uint64_t>(_mm_cvtsi128_si64(hi_shift));
    out[word_index] = low_word;
    out[word_index + 1] = high_word;
  }
  for (; word_index < word_count; ++word_index) {
    out[word_index] = lhs_words[word_index] ^ rhs_words[word_index];
  }
}

__attribute__((target("sse2"))) inline std::size_t HammingWords(std::span<const std::uint64_t> lhs_words,
                                                                std::span<const std::uint64_t> rhs_words) noexcept {
  static constexpr std::size_t kWordsPer128Bit = 2U;
  const std::size_t word_count = lhs_words.size();
  const std::size_t vector_loop_words = (word_count / kWordsPer128Bit) * kWordsPer128Bit;
  std::size_t total = 0;
  std::size_t word_index = 0;
  for (; word_index < vector_loop_words; word_index += kWordsPer128Bit) {
    __m128i vec_lhs = _mm_set_epi64x(static_cast<std::int64_t>(lhs_words[word_index + 1]),
                                     static_cast<std::int64_t>(lhs_words[word_index]));
    __m128i vec_rhs = _mm_set_epi64x(static_cast<std::int64_t>(rhs_words[word_index + 1]),
                                     static_cast<std::int64_t>(rhs_words[word_index]));
    __m128i vec_xor = _mm_xor_si128(vec_lhs, vec_rhs);
    const auto low_word = static_cast<std::uint64_t>(_mm_cvtsi128_si64(vec_xor));
    const __m128i hi_shift = _mm_srli_si128(vec_xor, 8);
    const auto high_word = static_cast<std::uint64_t>(_mm_cvtsi128_si64(hi_shift));
    total += Popcount64(low_word);
    total += Popcount64(high_word);
  }
  for (; word_index < word_count; ++word_index) {
    total += Popcount64(lhs_words[word_index] ^ rhs_words[word_index]);
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

// SSE2 implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
void BindSSE2(const core::HyperVector<Dim, bool>& lhs, const core::HyperVector<Dim, bool>& rhs,
              core::HyperVector<Dim, bool>* out) noexcept {
  const auto& lhs_words = lhs.Words();
  const auto& rhs_words = rhs.Words();
  auto& out_words = out->Words();
  BindWords(std::span<const std::uint64_t>(lhs_words),
            std::span<const std::uint64_t>(rhs_words),
            std::span<std::uint64_t>(out_words));
}

// SSE2 implementation of Hamming distance.
// Uses hardware popcount when available, scalar fallback otherwise.
template <std::size_t Dim>
[[nodiscard]] std::size_t HammingDistanceSSE2(const core::HyperVector<Dim, bool>& lhs,
                                              const core::HyperVector<Dim, bool>& rhs) noexcept {
  const auto& lhs_words = lhs.Words();
  const auto& rhs_words = rhs.Words();
  return HammingWords(std::span<const std::uint64_t>(lhs_words),
                      std::span<const std::uint64_t>(rhs_words));
}

}  // namespace hyperstream::backend::sse2

#endif  // x86/x64 guard
