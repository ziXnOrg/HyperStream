#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "hyperstream/backend/cpu_backend_avx2.hpp"

namespace hyperstream {
namespace backend {
namespace avx2 {

// TU-local AVX2 popcount used by MSVC build to avoid header intrinsic exposure under clang-tidy
inline std::uint64_t Popcount256(__m256i vector_value) noexcept {
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
  const std::uint64_t lo0 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(sad_lo));
  const std::uint64_t hi0 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(sad_lo, 8)));
  const std::uint64_t lo1 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(sad_hi));
  const std::uint64_t hi1 = static_cast<std::uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(sad_hi, 8)));
  return lo0 + hi0 + lo1 + hi1;
}

// MSVC TU: compile with /arch:AVX2. Implements raw AVX2 Hamming on 64-bit words.
std::size_t HammingWords(const std::uint64_t* input_a, const std::uint64_t* input_b, std::size_t word_count) noexcept {
  const std::size_t avx2_words = (word_count / 4) * 4;
  std::size_t total = 0;
  std::size_t word_index = 0;
  for (; word_index < avx2_words; word_index += 4) {
    __m256i vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_a[word_index]));
    __m256i vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_b[word_index]));
    __m256i vec_xor = _mm256_xor_si256(vec_a, vec_b);
    total += Popcount256(vec_xor);
  }
  for (; word_index < word_count; ++word_index) {
#if defined(_MSC_VER)
    total += __popcnt64(input_a[word_index] ^ input_b[word_index]);
#else
    // Fallback: builtins (should not be used for MSVC TU)
    total += __builtin_popcountll(input_a[word_index] ^ input_b[word_index]);
#endif
  }
  return total;
}

}  // namespace avx2
}  // namespace backend
}  // namespace hyperstream
#endif  // x86/x64 guard
