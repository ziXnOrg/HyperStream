#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

#include "hyperstream/backend/cpu_backend_sse2.hpp"

namespace hyperstream {
namespace backend {
namespace sse2 {

// MSVC TU: compile with /arch:SSE2. Implements raw SSE2 XOR on 64-bit words.
void BindWords(const std::uint64_t* input_a, const std::uint64_t* input_b, std::uint64_t* output,
               std::size_t word_count) noexcept {
  const std::size_t sse2_words = (word_count / 2) * 2;
  std::size_t word_index = 0;
  for (; word_index < sse2_words; word_index += 2) {
    __m128i vec_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input_a[word_index]));
    __m128i vec_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input_b[word_index]));
    __m128i vec_xor = _mm_xor_si128(vec_a, vec_b);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&output[word_index]), vec_xor);
  }
  for (; word_index < word_count; ++word_index) output[word_index] = input_a[word_index] ^ input_b[word_index];
}

}  // namespace sse2
}  // namespace backend
}  // namespace hyperstream
#endif  // x86/x64 guard
