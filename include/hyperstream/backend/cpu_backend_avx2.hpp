#pragma once

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

// Harley-Seal popcount for __m256i (256-bit vector).
// Uses CSA (Carry-Save Adder) approach to count bits in parallel.
inline std::uint64_t Popcount256(__m256i v) {
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

// AVX2 implementation of Bind (XOR) for binary hypervectors.
template <std::size_t Dim>
void BindAVX2(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
              core::HyperVector<Dim, bool>* out) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  auto& out_words = out->Words();

  const std::size_t num_words = a_words.size();
  const std::size_t avx2_words = (num_words / 4) * 4;  // Process 4 uint64_t at a time

  // Runtime heuristics for streaming stores to reduce cache pollution on large outputs.
  const std::size_t out_bytes = num_words * sizeof(std::uint64_t);
  const bool aligned32_in_a = ((reinterpret_cast<std::uintptr_t>(&a_words[0]) & 31u) == 0u);
  const bool aligned32_in_b = ((reinterpret_cast<std::uintptr_t>(&b_words[0]) & 31u) == 0u);
  const bool aligned32_out  = ((reinterpret_cast<std::uintptr_t>(&out_words[0]) & 31u) == 0u);
  const bool use_aligned_io = aligned32_in_a && aligned32_in_b;  // out handled per-store below

  // Stream stores only for very large transfers with aligned output (conservative)
  // Threshold can be overridden via env var HYPERSTREAM_NT_THRESHOLD_BYTES
  static const std::size_t nt_threshold = []() -> std::size_t {
    if (const char* env = std::getenv("HYPERSTREAM_NT_THRESHOLD_BYTES")) {
      char* end = nullptr;
      unsigned long long v = std::strtoull(env, &end, 10);
      if (end && *end == '\0' && v > 0ULL) return static_cast<std::size_t>(v);
    }
    return (1u << 20);  // default 1MB
  }();
  const bool use_stream = (out_bytes >= nt_threshold) && aligned32_out;
  (void)use_stream; // retained for future A/B, currently unused
  (void)nt_threshold;
  (void)use_aligned_io;

  std::size_t i = 0;
  // AVX2 path: single 256-bit chunk per iteration; no manual prefetch; storeu stores.
  for (; i < avx2_words; i += 4) {
    // A/B-4: force unaligned loads to remove alignment branch overhead for evaluation.
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a_words[i]));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_words[i]));
    __m256i vout = _mm256_xor_si256(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&out_words[i]), vout);
  }

  // Scalar tail for remaining words.
  for (; i < num_words; ++i) {
    out_words[i] = a_words[i] ^ b_words[i];
  }

  if (use_stream) {
    _mm_sfence();  // Ensure non-temporal stores are globally visible before returning
  }
}

// AVX2 implementation of Hamming distance.
template <std::size_t Dim>
std::size_t HammingDistanceAVX2(const core::HyperVector<Dim, bool>& a,
                                const core::HyperVector<Dim, bool>& b) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();

  const std::size_t num_words = a_words.size();
  const std::size_t avx2_words = (num_words / 4) * 4;

  std::size_t total = 0;
  std::size_t i = 0;

  // AVX2 path: XOR + popcount 4 uint64_t at a time.
  for (; i < avx2_words; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a_words[i]));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b_words[i]));
    __m256i vxor = _mm256_xor_si256(va, vb);
    total += Popcount256(vxor);
  }

  // Scalar tail.
  for (; i < num_words; ++i) {
    const std::uint64_t xor_word = a_words[i] ^ b_words[i];
#if defined(__GNUC__) || defined(__clang__)
    total += __builtin_popcountll(xor_word);
#elif defined(_MSC_VER)
    total += __popcnt64(xor_word);
#else
    // Fallback: Kernighan's method.
    std::uint64_t x = xor_word;
    while (x) {
      x &= (x - 1);
      ++total;
    }
#endif
  }

  return total;
}

}  // namespace avx2
}  // namespace backend
}  // namespace hyperstream
