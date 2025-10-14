#pragma once

// =============================================================================
// File:        include/hyperstream/backend/cpu_backend_neon.hpp
// Overview:    NEON-accelerated primitives (Bind, Hamming) for AArch64.
// Mathematical Foundation: XOR-bind; Hamming via per-byte popcount and sum.
// Security Considerations: Unaligned loads/stores; contiguous word layout;
//              noexcept functions.
// Performance Considerations: 128-bit vector ops; scalar tail handling; uses
//              vcnt and vaddvq for efficient byte popcount and reduction.
// Examples:    See backend/cpu_backend.hpp for dispatch usage.
// =============================================================================
#if defined(__aarch64__) || defined(_M_ARM64)

// NEON-accelerated backend primitives for AArch64 platforms (ARMv8+ Advanced SIMD).
// Implements Bind (XOR) and Hamming distance using 128-bit NEON operations.
// Invariants and I/O contract follow SSE2/AVX2 backends:
// - Unaligned memory semantics: vld1q_u64/vst1q_u64 (unaligned allowed on AArch64).
// - Contiguous word layout: operate over HyperVector<Dim,bool>::Words() (uint64_t[]).
// - Safe tail handling: scalar tail loop completes remainder; final-word mask is handled by
// callers.
// - Compiler targets: NEON is mandatory on AArch64; no special function attributes required.

#include <arm_neon.h>
#include <cstddef>
#include <cstdint>
#include <span>
#include <bit>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream::backend::neon {

/// Portable 64-bit popcount helper (C++20 first, builtin fallback).
inline std::size_t Popcount64(std::uint64_t value) noexcept {
#if defined(__cpp_lib_bitops) && (__cpp_lib_bitops >= 201907L)
  return static_cast<std::size_t>(std::popcount(value));
#else
  return static_cast<std::size_t>(__builtin_popcountll(value));
#endif
}

/// XOR-bind two arrays of 64-bit words using NEON with unaligned IO (span-based primary API).
inline void BindWords(std::span<const std::uint64_t> lhs_words,
                      std::span<const std::uint64_t> rhs_words,
                      std::span<std::uint64_t> out) noexcept {
  // Safety: AArch64 vld1q_u64 supports unaligned loads. We still bound the vector loop
  // to the minimum of the provided spans to avoid any OOB when callers pass mismatched sizes.
  static constexpr std::size_t kWordsPer128Bit = 2U;  // 2x u64 per 128-bit lane
  const std::size_t n12 = (lhs_words.size() < rhs_words.size()) ? lhs_words.size() : rhs_words.size();
  const std::size_t n = (n12 < out.size()) ? n12 : out.size();
  const std::size_t vector_loop_words = (n / kWordsPer128Bit) * kWordsPer128Bit;

  std::size_t i = 0;
  for (; i < vector_loop_words; i += kWordsPer128Bit) {
    const uint64x2_t vec_lhs = vld1q_u64(lhs_words.data() + i);
    const uint64x2_t vec_rhs = vld1q_u64(rhs_words.data() + i);
    const uint64x2_t vec_xor = veorq_u64(vec_lhs, vec_rhs);
    vst1q_u64(out.data() + i, vec_xor);
  }
  // Scalar tail handles any remainder words (including n not multiple of 2).
  for (; i < n; ++i) {
    out[i] = lhs_words[i] ^ rhs_words[i];
  }
}

/// Pointer-based forwarder for legacy call sites.
inline void BindWords(const std::uint64_t* lhs_words, const std::uint64_t* rhs_words,
                      std::uint64_t* out, std::size_t word_count) noexcept {
  BindWords(std::span<const std::uint64_t>(lhs_words, word_count),
            std::span<const std::uint64_t>(rhs_words, word_count),
            std::span<std::uint64_t>(out, word_count));
}

/// Compute Hamming distance between two word arrays using NEON (span-based primary API).
inline std::size_t HammingWords(std::span<const std::uint64_t> lhs_words,
                                std::span<const std::uint64_t> rhs_words) noexcept {
  // Safety: bound the loop by the minimum span size to avoid OOB on mismatched inputs.
  // Tail path uses scalar popcount for any remaining word.
  // Use widening pairwise reduction instead of vaddvq_u8 for better platform compatibility.
  static constexpr std::size_t kWordsPer128Bit = 2U;
  const std::size_t n = (lhs_words.size() < rhs_words.size()) ? lhs_words.size() : rhs_words.size();
  const std::size_t vector_loop_words = (n / kWordsPer128Bit) * kWordsPer128Bit;

  std::size_t total = 0;
  std::size_t i = 0;
  for (; i < vector_loop_words; i += kWordsPer128Bit) {
    const uint64x2_t vec_lhs = vld1q_u64(lhs_words.data() + i);
    const uint64x2_t vec_rhs = vld1q_u64(rhs_words.data() + i);
    const uint64x2_t vec_xor = veorq_u64(vec_lhs, vec_rhs);
    const uint8x16_t xor_bytes = vreinterpretq_u8_u64(vec_xor);
    const uint8x16_t popcnt_bytes = vcntq_u8(xor_bytes);

    // Widening pairwise reduction: u8 -> u16 -> u32 -> u64 -> horizontal add
    // This is more reliable across NEON implementations than vaddvq_u8.
    const uint16x8_t sum16 = vpaddlq_u8(popcnt_bytes);   // 16x u8 -> 8x u16
    const uint32x4_t sum32 = vpaddlq_u16(sum16);         // 8x u16 -> 4x u32
    const uint64x2_t sum64 = vpaddlq_u32(sum32);         // 4x u32 -> 2x u64
    const std::uint64_t lane0 = vgetq_lane_u64(sum64, 0);
    const std::uint64_t lane1 = vgetq_lane_u64(sum64, 1);
    total += static_cast<std::size_t>(lane0 + lane1);
  }
  for (; i < n; ++i) {
    total += Popcount64(lhs_words[i] ^ rhs_words[i]);
  }
  return total;
}

/// Pointer-based forwarder for legacy call sites.
inline std::size_t HammingWords(const std::uint64_t* lhs_words, const std::uint64_t* rhs_words,
                                std::size_t word_count) noexcept {
  return HammingWords(std::span<const std::uint64_t>(lhs_words, word_count),
                      std::span<const std::uint64_t>(rhs_words, word_count));
}

// NEON implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
inline void BindNEON(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
                     core::HyperVector<Dim, bool>* out) noexcept {
  const auto& lhs_words = a.Words();
  const auto& rhs_words = b.Words();
  auto& out_words = out->Words();
  BindWords(lhs_words, rhs_words, out_words);
}

// NEON implementation of Hamming distance.
template <std::size_t Dim>
inline std::size_t HammingDistanceNEON(const core::HyperVector<Dim, bool>& a,
                                       const core::HyperVector<Dim, bool>& b) noexcept {
  const auto& lhs_words = a.Words();
  const auto& rhs_words = b.Words();
  return HammingWords(lhs_words, rhs_words);
}

}  // namespace hyperstream::backend::neon

#endif  // AArch64 guard
