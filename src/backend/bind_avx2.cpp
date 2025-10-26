#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cstddef>
#include <cstdint>
#include <immintrin.h>

#include "hyperstream/backend/cpu_backend_avx2.hpp"

namespace hyperstream {
namespace backend {
namespace avx2 {

// MSVC TU: compile with /arch:AVX2. Implements raw AVX2 XOR on 64-bit words.
void BindWords(const std::uint64_t* input_a, const std::uint64_t* input_b, std::uint64_t* output,
               std::size_t word_count) noexcept {
  const std::size_t avx2_words = (word_count / 4) * 4;
  std::size_t word_index = 0;
  for (; word_index < avx2_words; word_index += 4) {
    __m256i vec_a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_a[word_index]));
    __m256i vec_b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&input_b[word_index]));
    __m256i vec_xor = _mm256_xor_si256(vec_a, vec_b);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&output[word_index]), vec_xor);
  }
  for (; word_index < word_count; ++word_index) output[word_index] = input_a[word_index] ^ input_b[word_index];
}

}  // namespace avx2
}  // namespace backend
}  // namespace hyperstream
#endif  // x86/x64 guard
