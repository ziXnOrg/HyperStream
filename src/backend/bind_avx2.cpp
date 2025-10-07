#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cstddef>
#include <cstdint>
#include <immintrin.h>
#include "hyperstream/backend/cpu_backend_avx2.hpp"

namespace hyperstream { namespace backend { namespace avx2 {

// MSVC TU: compile with /arch:AVX2. Implements raw AVX2 XOR on 64-bit words.
void BindWords(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t word_count) {
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

}}} // namespace hyperstream::backend::avx2
#endif // x86/x64 guard


