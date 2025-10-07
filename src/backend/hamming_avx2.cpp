#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "hyperstream/backend/cpu_backend_avx2.hpp"

namespace hyperstream { namespace backend { namespace avx2 {

// MSVC TU: compile with /arch:AVX2. Implements raw AVX2 Hamming on 64-bit words.
std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b, std::size_t word_count) {
  const std::size_t avx2_words = (word_count / 4) * 4;
  std::size_t total = 0; std::size_t i = 0;
  for (; i < avx2_words; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&a[i]));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&b[i]));
    __m256i vx = _mm256_xor_si256(va, vb);
    total += Popcount256(vx);
  }
  for (; i < word_count; ++i) {
#if defined(_MSC_VER)
    total += __popcnt64(a[i] ^ b[i]);
#else
    // Fallback: builtins (should not be used for MSVC TU)
    total += __builtin_popcountll(a[i] ^ b[i]);
#endif
  }
  return total;
}

}}} // namespace hyperstream::backend::avx2
#endif // x86/x64 guard


