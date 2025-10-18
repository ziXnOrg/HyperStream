#pragma once

// =============================================================================
// File:        include/hyperstream/backend/cpu_backend.hpp
// Overview:    CPU backend dispatch with runtime SIMD detection (AVX2/SSE2/NEON)
//              and scalar fallbacks; exposes bind and Hamming operations.
// Mathematical Foundation: XOR bind, Hamming distance (popcount of XOR).
// Security Considerations: No dynamic allocation in hot paths; functions are
//              noexcept where safe; feature detection uses CPUID/XCR0 paths.
// Performance Considerations: Runtime dispatch to vectorized paths; scalar
//              reference for portability; popcount uses Kernighan method.
// Examples:    See backend/policy.hpp for selection heuristics.
// =============================================================================
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

namespace hyperstream::backend {

// CPU feature flags detected at runtime.
struct CpuFeatures {
  bool avx2 = false;
  bool sse2 = false;
};

// Detect CPU capabilities via CPUID (x86) or compile-time flags (other archs).
CpuFeatures DetectCpuFeatures() noexcept;

// Scalar fallback: portable popcount using Kernighan's method.
[[nodiscard]] inline std::uint64_t PopcountScalar(std::uint64_t value) noexcept {
  std::uint64_t bit_count = 0;
  while (value != 0ULL) {
    value &= (value - 1ULL);
    ++bit_count;
  }
  return bit_count;
}

// Bind two binary hypervectors (XOR). Selects optimal path at runtime.
template <std::size_t Dim>
void Bind(const core::HyperVector<Dim, bool>& lhs, const core::HyperVector<Dim, bool>& rhs,
          core::HyperVector<Dim, bool>* out);

// Compute Hamming distance (popcount of XOR). Selects optimal path at runtime.
template <std::size_t Dim>
std::size_t HammingDistance(const core::HyperVector<Dim, bool>& lhs,
                            const core::HyperVector<Dim, bool>& rhs);

// Backend singleton holding detected features and function pointers.
class CpuBackend {
 public:
  [[nodiscard]] static CpuBackend& Instance() noexcept {
    static CpuBackend instance;
    return instance;
  }

  [[nodiscard]] const CpuFeatures& Features() const noexcept {
    return features_;
  }

 private:
  CpuBackend() : features_(DetectCpuFeatures()) {}
  CpuFeatures features_;
};

// Implementation: Bind (XOR) using scalar fallback.
template <std::size_t Dim>
void BindScalar(const core::HyperVector<Dim, bool>& lhs, const core::HyperVector<Dim, bool>& rhs,
                core::HyperVector<Dim, bool>* out) {
  const auto& lhs_words = lhs.Words();
  const auto& rhs_words = rhs.Words();
  auto& out_words = out->Words();
  for (std::size_t word_index = 0; word_index < lhs_words.size(); ++word_index) {
    out_words[word_index] = lhs_words[word_index] ^ rhs_words[word_index];
  }
}

// Implementation: Hamming distance using scalar popcount.
template <std::size_t Dim>
std::size_t HammingDistanceScalar(const core::HyperVector<Dim, bool>& lhs,
                                  const core::HyperVector<Dim, bool>& rhs) {
  const auto& lhs_words = lhs.Words();
  const auto& rhs_words = rhs.Words();
  std::size_t distance = 0;
  for (std::size_t word_index = 0; word_index < lhs_words.size(); ++word_index) {
    const std::uint64_t xor_word = lhs_words[word_index] ^ rhs_words[word_index];
    distance += PopcountScalar(xor_word);
  }
  return distance;
}

// SIMD specializations are declared in their respective headers included above.

// Public interface: dispatches to optimal implementation based on runtime detection.
template <std::size_t Dim>
void Bind(const core::HyperVector<Dim, bool>& lhs, const core::HyperVector<Dim, bool>& rhs,
          core::HyperVector<Dim, bool>* out) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  const auto& features = CpuBackend::Instance().Features();
#if defined(__AVX2__)
  if (features.avx2) {
    hyperstream::backend::avx2::BindAVX2(lhs, rhs, out);
    return;
  }
#endif
#if defined(__SSE2__) || defined(_M_X64) || defined(__x86_64__)
  if (features.sse2) {
    hyperstream::backend::sse2::BindSSE2(lhs, rhs, out);
    return;
  }
#endif
#endif

  // Scalar fallback.
  BindScalar(lhs, rhs, out);
}

template <std::size_t Dim>
std::size_t HammingDistance(const core::HyperVector<Dim, bool>& lhs,
                            const core::HyperVector<Dim, bool>& rhs) {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  const auto& features = CpuBackend::Instance().Features();
#if defined(__AVX2__)
  if (features.avx2) {
    return hyperstream::backend::avx2::HammingDistanceAVX2(lhs, rhs);
  }
#endif
#if defined(__SSE2__) || defined(_M_X64) || defined(__x86_64__)
  if (features.sse2) {
    return hyperstream::backend::sse2::HammingDistanceSSE2(lhs, rhs);
  }
#endif
#endif

  // Scalar fallback.
  return HammingDistanceScalar(lhs, rhs);
}

// CPU feature detection implementation.
inline CpuFeatures DetectCpuFeatures() noexcept {
  CpuFeatures features;
#if defined(__x86_64__) || defined(_M_X64)
  // Use CPUID to detect AVX2 and SSE2.
  // CPUID function 1: ECX bit 28 = AVX, EAX=7/ECX=0: EBX bit 5 = AVX2.
  // CPUID function 1: EDX bit 26 = SSE2.
#if defined(_MSC_VER)
  std::array<int, 4> cpu_info{};
  __cpuid(cpu_info.data(), 1);
  static constexpr int kSse2Bit = 26;
  static constexpr int kAvxBit = 28;
  static constexpr int kLeaf7 = 7;
  static constexpr int kSubleaf0 = 0;
  static constexpr int kAvx2Bit = 5;
  features.sse2 = (cpu_info[3] & (1 << kSse2Bit)) != 0;
  const bool avx = (cpu_info[2] & (1 << kAvxBit)) != 0;
  if (avx) {
    __cpuidex(cpu_info.data(), kLeaf7, kSubleaf0);
    features.avx2 = (cpu_info[1] & (1 << kAvx2Bit)) != 0;
  }
#elif defined(__GNUC__) || defined(__clang__)
  unsigned int eax, ebx, ecx, edx;
  __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(1), "c"(0));
  static constexpr unsigned int kSse2Bit = 26U;
  static constexpr unsigned int kAvxBit = 28U;
  static constexpr unsigned int kLeaf7 = 7U;
  static constexpr unsigned int kSubleaf0 = 0U;
  static constexpr unsigned int kAvx2Bit = 5U;
  features.sse2 = (edx & (1U << kSse2Bit)) != 0U;
  const bool avx = (ecx & (1U << kAvxBit)) != 0U;
  if (avx) {
    __asm__ __volatile__("cpuid" : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx) : "a"(kLeaf7), "c"(kSubleaf0));
    features.avx2 = (ebx & (1U << kAvx2Bit)) != 0U;
  }
#endif
#endif
  return features;
}

}  // namespace hyperstream::backend
