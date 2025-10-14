#pragma once

// CPU capability detection (x86/x86_64) with safe fallbacks for non-x86.
// Provides minimal feature mask API used by the backend policy layer.

#include <cstdint>
#include <array>

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include <cpuid.h>
#endif
#endif

namespace hyperstream::backend {

enum class CpuFeature : std::uint8_t {
  SSE2 = 0x1U,
  AVX2 = 0x2U,
  NEON = 0x4U,
};

[[nodiscard]] inline auto HasFeature(std::uint32_t mask, CpuFeature feature) noexcept -> bool {
  return (mask & static_cast<std::uint32_t>(feature)) != 0U;
}

// Platform-neutral wrappers. Return false on non-x86 or when probing is unavailable.
[[nodiscard]] inline auto DetectSSE2() noexcept -> bool {
#if defined(__x86_64__) || defined(_M_X64)
  // SSE2 is baseline on x86_64
  return true;
#elif defined(_MSC_VER) && (defined(_M_IX86))
  std::array<int, 4> regs = {0, 0, 0, 0};
  __cpuid(regs.data(), 1);
  // EDX bit 26 indicates SSE2
  return (static_cast<unsigned>(regs[3]) & (1U << 26U)) != 0U;
#elif (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  return (edx & (1U << 26U)) != 0U;  // SSE2
#else
  return false;
#endif
}

[[nodiscard]] inline auto ReadXcr0() noexcept -> std::uint64_t {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  return _xgetbv(0);
#elif (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
  unsigned int eax, edx;
  __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
  return 0;
#endif
}

[[nodiscard]] inline auto DetectAVX2() noexcept -> bool {
#if !(defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64))
  return false;
#else
  // Step 1: CPUID.1:ECX OSXSAVE and AVX
#if defined(_MSC_VER)
  std::array<int, 4> regs{};
  __cpuid(regs.data(), 1);
  auto ecx = static_cast<unsigned int>(regs[2]);
#elif defined(__GNUC__) || defined(__clang__)
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
#endif
  static constexpr unsigned kOsxsaveBit = 27U;
  static constexpr unsigned kAvxBit = 28U;
  const bool osxsave = (ecx & (1U << kOsxsaveBit)) != 0U;
  const bool avx = (ecx & (1U << kAvxBit)) != 0U;
  if (!(osxsave && avx)) {
    return false;
  }

  // Step 2: XCR0 must enable YMM state (bits 1 and 2)
  const std::uint64_t xcr0 = ReadXcr0();
  static constexpr std::uint64_t kXcr0YmmMask = 0x6ULL;
  const bool ymm_enabled = ((xcr0 & kXcr0YmmMask) == kXcr0YmmMask);
  if (!ymm_enabled) {
    return false;
  }

  // Step 3: CPUID.7.0:EBX bit 5 indicates AVX2
#if defined(_MSC_VER)
  std::array<int, 4> regs7{};
  static constexpr int kCpuidLeaf7 = 7;
  static constexpr int kCpuidSubleaf0 = 0;
  __cpuidex(regs7.data(), kCpuidLeaf7, kCpuidSubleaf0);
  const auto ebx_val = static_cast<unsigned int>(regs7[1]);
#elif defined(__GNUC__) || defined(__clang__)
  unsigned int eax7, ecx7, edx7, ebx_val;
  if (!__get_cpuid_count(7, 0, &eax7, &ebx_val, &ecx7, &edx7)) return false;
#endif
  static constexpr unsigned kAvx2Bit = 5U;
  return (ebx_val & (1U << kAvx2Bit)) != 0U;
#endif
}

[[nodiscard]] inline auto DetectNEON() noexcept -> bool {
#if defined(__aarch64__) || defined(_M_ARM64)
  // Advanced SIMD is mandatory in AArch64
  return true;
#else
  return false;
#endif
}

[[nodiscard]] inline auto GetCpuFeatureMask() noexcept -> std::uint32_t {
#if defined(HYPERSTREAM_FORCE_SCALAR)
  return 0U;
#else
  std::uint32_t mask = 0U;
  if (DetectSSE2()) {
    mask |= static_cast<std::uint32_t>(CpuFeature::SSE2);
  }
  if (DetectAVX2()) {
    mask |= static_cast<std::uint32_t>(CpuFeature::AVX2);
  }
  if (DetectNEON()) {
    mask |= static_cast<std::uint32_t>(CpuFeature::NEON);
  }
  return mask;
#endif
}

}  // namespace hyperstream::backend
