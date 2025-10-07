#include <cstddef>
#include <cstdint>
#include <emmintrin.h>
#include "hyperstream/backend/cpu_backend_sse2.hpp"

namespace hyperstream { namespace backend { namespace sse2 {

// MSVC TU: compile with /arch:SSE2. Implements raw SSE2 XOR on 64-bit words.
void BindWords(const std::uint64_t* a, const std::uint64_t* b, std::uint64_t* out, std::size_t word_count) {
  const std::size_t sse2_words = (word_count / 2) * 2;
  std::size_t i = 0;
  for (; i < sse2_words; i += 2) {
    __m128i va = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&a[i]));
    __m128i vb = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&b[i]));
    __m128i vx = _mm_xor_si128(va, vb);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&out[i]), vx);
  }
  for (; i < word_count; ++i) out[i] = a[i] ^ b[i];
}

}}} // namespace hyperstream::backend::sse2

