#pragma once

// CPU backend with runtime SIMD detection and fallback.
// Detects AVX2/SSE on x86-64 at initialization; dispatches binding, popcount, and
// bundling to optimal code path. Scalar fallback ensures portability. Header-only.

#include <array>
#include <cstddef>
#include <cstdint>

#include "hyperstream/core/hypervector.hpp"

#if defined(__AVX2__)
#include "hyperstream/backend/cpu_backend_avx2.hpp"
#endif

#if defined(__SSE2__) || defined(_M_X64) || defined(__x86_64__)
#include "hyperstream/backend/cpu_backend_sse2.hpp"
#endif

namespace hyperstream {
namespace backend {

// CPU feature flags detected at runtime.
struct CpuFeatures {
  bool avx2 = false;
  bool sse2 = false;
};

// Detect CPU capabilities via CPUID (x86) or compile-time flags (other archs).
CpuFeatures DetectCpuFeatures();

// Scalar fallback: portable popcount using Kernighan's method.
inline std::uint64_t PopcountScalar(std::uint64_t x) {
  std::uint64_t count = 0;
  while (x) {
    x &= (x - 1);
    ++count;
  }
  return count;
}

// Bind two binary hypervectors (XOR). Selects optimal path at runtime.
template <std::size_t Dim>
void Bind(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
          core::HyperVector<Dim, bool>* out);

// Compute Hamming distance (popcount of XOR). Selects optimal path at runtime.
template <std::size_t Dim>
std::size_t HammingDistance(const core::HyperVector<Dim, bool>& a,
                            const core::HyperVector<Dim, bool>& b);

// Backend singleton holding detected features and function pointers.
class CpuBackend {
 public:
  static CpuBackend& Instance() {
    static CpuBackend instance;
    return instance;
  }

  const CpuFeatures& Features() const {
    return features_;
  }

 private:
  CpuBackend() : features_(DetectCpuFeatures()) {}
  CpuFeatures features_;
};

// Implementation: Bind (XOR) using scalar fallback.
template <std::size_t Dim>
void BindScalar(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
                core::HyperVector<Dim, bool>* out) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  auto& out_words = out->Words();
  for (std::size_t i = 0; i < a_words.size(); ++i) {
    out_words[i] = a_words[i] ^ b_words[i];
  }
}

// Implementation: Hamming distance using scalar popcount.
template <std::size_t Dim>
std::size_t HammingDistanceScalar(const core::HyperVector<Dim, bool>& a,
                                  const core::HyperVector<Dim, bool>& b) {
  const auto& a_words = a.Words();
  const auto& b_words = b.Words();
  std::size_t dist = 0;
  for (std::size_t i = 0; i < a_words.size(); ++i) {
    const std::uint64_t xor_word = a_words[i] ^ b_words[i];
    dist += PopcountScalar(xor_word);
  }
  return dist;
}

// Forward declare SIMD namespaces and functions (relative to current namespace)
namespace avx2 {
template <size_t Dim> void BindAVX2(
    const core::HyperVector<Dim, bool>& a,
    const core::HyperVector<Dim, bool>& b,
    core::HyperVector<Dim, bool>* out);

template <size_t Dim> size_t HammingDistanceAVX2(
    const core::HyperVector<Dim, bool>& a,
    const core::HyperVector<Dim, bool>& b);
}  // namespace avx2

namespace sse2 {
template <size_t Dim> void BindSSE2(
    const core::HyperVector<Dim, bool>& a,
    const core::HyperVector<Dim, bool>& b,
    core::HyperVector<Dim, bool>* out);

template <size_t Dim> size_t HammingDistanceSSE2(
    const core::HyperVector<Dim, bool>& a,
    const core::HyperVector<Dim, bool>& b);
}  // namespace sse2


// Public interface: dispatches to optimal implementation based on runtime detection.
template <std::size_t Dim>
void Bind(const core::HyperVector<Dim, bool>& a, const core::HyperVector<Dim, bool>& b,
          core::HyperVector<Dim, bool>* out) {
  const auto& features = CpuBackend::Instance().Features();

#if defined(__AVX2__)
  if (features.avx2) {
    hyperstream::backend::avx2::BindAVX2(a, b, out);
    return;
  }
#endif

#if defined(__SSE2__) || defined(_M_X64) || defined(__x86_64__)
  if (features.sse2) {
    hyperstream::backend::sse2::BindSSE2(a, b, out);
    return;
  }
#endif

  // Scalar fallback.
  BindScalar(a, b, out);
}

template <std::size_t Dim>
std::size_t HammingDistance(const core::HyperVector<Dim, bool>& a,
                            const core::HyperVector<Dim, bool>& b) {
  const auto& features = CpuBackend::Instance().Features();

#if defined(__AVX2__)
  if (features.avx2) {
    return hyperstream::backend::avx2::HammingDistanceAVX2(a, b);
  }
#endif

#if defined(__SSE2__) || defined(_M_X64) || defined(__x86_64__)
  if (features.sse2) {
    return hyperstream::backend::sse2::HammingDistanceSSE2(a, b);
  }
#endif

  // Scalar fallback.
  return HammingDistanceScalar(a, b);
}

// CPU feature detection implementation.
inline CpuFeatures DetectCpuFeatures() {
  CpuFeatures features;
#if defined(__x86_64__) || defined(_M_X64)
  // Use CPUID to detect AVX2 and SSE2.
  // CPUID function 1: ECX bit 28 = AVX, EAX=7/ECX=0: EBX bit 5 = AVX2.
  // CPUID function 1: EDX bit 26 = SSE2.
#if defined(_MSC_VER)
  int cpu_info[4];
  __cpuid(cpu_info, 1);
  features.sse2 = (cpu_info[3] & (1 << 26)) != 0;
  const bool avx = (cpu_info[2] & (1 << 28)) != 0;
  if (avx) {
    __cpuidex(cpu_info, 7, 0);
    features.avx2 = (cpu_info[1] & (1 << 5)) != 0;
  }
#elif defined(__GNUC__) || defined(__clang__)
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
  features.sse2 = (edx & (1 << 26)) != 0;
  const bool avx = (ecx & (1 << 28)) != 0;
  if (avx) {
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(7), "c"(0));
    features.avx2 = (ebx & (1 << 5)) != 0;
  }
#endif
#endif
  return features;
}

}  // namespace backend
}  // namespace hyperstream
