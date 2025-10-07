#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include "hyperstream/backend/cpu_backend_sse2.hpp"

namespace hyperstream { namespace backend { namespace sse2 {

// MSVC TU: compile with /arch:SSE2. Implements raw SSE2 Hamming on 64-bit words.
std::size_t HammingWords(const std::uint64_t* a, const std::uint64_t* b, std::size_t word_count) {
  const std::size_t sse2_words = (word_count / 2) * 2;
  std::size_t total = 0; std::size_t i = 0;
  for (; i < sse2_words; i += 2) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
    __m128i vx = _mm_xor_si128(va, vb);
    std::uint64_t words[2];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(words), vx);
#if defined(_MSC_VER)
    total += __popcnt64(words[0]);
    total += __popcnt64(words[1]);
#else
    // Fallback: portable popcount (should not be used for MSVC TU)
    for (int j = 0; j < 2; ++j) {
      std::uint64_t x = words[j];
      while (x) { x &= (x - 1); ++total; }
    }
#endif
  }
  for (; i < word_count; ++i) {
#if defined(_MSC_VER)
    total += __popcnt64(a[i] ^ b[i]);
#else
    std::uint64_t x = a[i] ^ b[i];
    while (x) { x &= (x - 1); ++total; }
#endif
  }
  return total;
}

}}} // namespace hyperstream::backend::sse2

