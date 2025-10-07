#pragma once

#if defined(__aarch64__) || defined(_M_ARM64)

// NEON-accelerated backend primitives for AArch64 platforms (ARMv8+ Advanced SIMD).
// Implements Bind (XOR) and Hamming distance using 128-bit NEON operations.
// Invariants and I/O contract follow SSE2/AVX2 backends:
// - Unaligned memory semantics: vld1q_u64/vst1q_u64 (unaligned allowed on AArch64).
// - Contiguous word layout: operate over HyperVector<Dim,bool>::Words() (uint64_t[]).
// - Safe tail handling: scalar tail loop completes remainder; final-word mask is handled by callers.
// - Compiler targets: NEON is mandatory on AArch64; no special function attributes required.

#include <cstddef>
#include <cstdint>
#include <arm_neon.h>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream {
namespace backend {
namespace neon {

/// XOR-bind two arrays of 64-bit words using NEON with unaligned IO.
inline void BindWords(const std::uint64_t* a, const std::uint64_t* b,
                      std::uint64_t* out, std::size_t word_count) {
  const std::size_t neon_words = (word_count / 2) * 2; // 2x u64 per 128-bit lane
  std::size_t i = 0;
  for (; i < neon_words; i += 2) {
    uint64x2_t va = vld1q_u64(reinterpret_cast<const uint64_t*>(&a[i]));
    uint64x2_t vb = vld1q_u64(reinterpret_cast<const uint64_t*>(&b[i]));
    uint64x2_t vx = veorq_u64(va, vb);
    vst1q_u64(reinterpret_cast<uint64_t*>(&out[i]), vx);
  }
  for (; i < word_count; ++i) out[i] = a[i] ^ b[i];
}

/// Compute Hamming distance between two word arrays using NEON.
inline std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b,
                                std::size_t word_count) {
  const std::size_t neon_words = (word_count / 2) * 2;
  std::size_t total = 0; std::size_t i = 0;
  for (; i < neon_words; i += 2) {
    // XOR
    uint64x2_t va = vld1q_u64(reinterpret_cast<const uint64_t*>(&a[i]));
    uint64x2_t vb = vld1q_u64(reinterpret_cast<const uint64_t*>(&b[i]));
    uint64x2_t vx = veorq_u64(va, vb);
    // Byte popcount then horizontal sum
    uint8x16_t bytes = vreinterpretq_u8_u64(vx);
    uint8x16_t pc = vcntq_u8(bytes);
    // vaddvq_u8 returns the sum of all 16 lanes (fits into <= 128)
    unsigned int sum = vaddvq_u8(pc);
    total += static_cast<std::size_t>(sum);
  }
  for (; i < word_count; ++i) total += static_cast<std::size_t>(__builtin_popcountll(a[i] ^ b[i]));
  return total;
}

// NEON implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
inline void BindNEON(const core::HyperVector<Dim, bool>& a,
                     const core::HyperVector<Dim, bool>& b,
                     core::HyperVector<Dim, bool>* out) {
  const auto& aw = a.Words();
  const auto& bw = b.Words();
  auto& ow = out->Words();
  BindWords(aw.data(), bw.data(), ow.data(), aw.size());
}

// NEON implementation of Hamming distance.
template <std::size_t Dim>
inline std::size_t HammingDistanceNEON(const core::HyperVector<Dim, bool>& a,
                                       const core::HyperVector<Dim, bool>& b) {
  const auto& aw = a.Words();
  const auto& bw = b.Words();
  return HammingWords(aw.data(), bw.data(), aw.size());
}

} // namespace neon
} // namespace backend
} // namespace hyperstream

#endif // AArch64 guard

