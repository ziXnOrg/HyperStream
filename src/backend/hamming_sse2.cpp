#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <cstddef>
#include <cstdint>
#include <emmintrin.h>

#include "hyperstream/backend/cpu_backend_sse2.hpp"

namespace hyperstream {
namespace backend {
namespace sse2 {

// MSVC TU: compile with /arch:SSE2. Implements raw SSE2 Hamming on 64-bit words.
std::size_t HammingWords(const std::uint64_t* input_a, const std::uint64_t* input_b, std::size_t word_count) noexcept {
  const std::size_t sse2_words = (word_count / 2) * 2;
  std::size_t total = 0;
  std::size_t word_index = 0;
  for (; word_index < sse2_words; word_index += 2) {
    __m128i vec_a = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input_a[word_index]));
    __m128i vec_b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&input_b[word_index]));
    __m128i vec_xor = _mm_xor_si128(vec_a, vec_b);
    std::uint64_t words[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(words), vec_xor);
#if defined(_MSC_VER)
    total += __popcnt64(words[0]);
    total += __popcnt64(words[1]);
#else
    // Fallback: portable popcount (should not be used for MSVC TU)
    for (int element_index = 0; element_index < 2; ++element_index) {
      std::uint64_t value = words[element_index];
      while (value) {
        value &= (value - 1);
        ++total;
      }
    }
#endif
  }
  for (; word_index < word_count; ++word_index) {
#if defined(_MSC_VER)
    total += __popcnt64(input_a[word_index] ^ input_b[word_index]);
#else
    std::uint64_t value = input_a[word_index] ^ input_b[word_index];
    while (value) {
      value &= (value - 1);
      ++total;
    }
#endif
  }
  return total;
}

}  // namespace sse2
}  // namespace backend
}  // namespace hyperstream
#endif  // x86/x64 guard
