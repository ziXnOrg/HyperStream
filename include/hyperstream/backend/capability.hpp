#pragma once

// CPU capability detection (x86/x86_64) with safe fallbacks for non-x86.
// Provides minimal feature mask API used by the backend policy layer.

#include <cstdint>

#if defined(_MSC_VER)
  #include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
  #if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
    #include <cpuid.h>
  #endif
#endif

namespace hyperstream {
namespace backend {

enum class CpuFeature : std::uint32_t {
  SSE2 = 0x1,
  AVX2 = 0x2,
  NEON = 0x4,
};

inline bool HasFeature(std::uint32_t mask, CpuFeature f) {
  return (mask & static_cast<std::uint32_t>(f)) != 0u;
}

// Platform-neutral wrappers. Return false on non-x86 or when probing is unavailable.
inline bool DetectSSE2() {
#if defined(__x86_64__) || defined(_M_X64)
  // SSE2 is baseline on x86_64
  return true;
#elif defined(_MSC_VER) && (defined(_M_IX86))
  int regs[4] = {0};
  __cpuid(regs, 1);
  // EDX bit 26 indicates SSE2
  return (regs[3] & (1 << 26)) != 0;
#elif (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
  unsigned int eax, ebx, ecx, edx;
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
  return (edx & (1u << 26)) != 0u; // SSE2
#else
  return false;
#endif
}

inline std::uint64_t xgetbv_xcr0() {
#if defined(_MSC_VER) && (defined(_M_IX86) || defined(_M_X64))
  return _xgetbv(0);
#elif (defined(__i386__) || defined(__x86_64__)) && (defined(__GNUC__) || defined(__clang__))
  unsigned int eax, edx;
  __asm__ volatile ("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
  return (static_cast<std::uint64_t>(edx) << 32) | eax;
#else
  return 0;
#endif
}

inline bool DetectAVX2() {
#if !(defined(__i386__) || defined(__x86_64__) || defined(_M_IX86) || defined(_M_X64))
  return false;
#else
  // Step 1: CPUID.1:ECX OSXSAVE and AVX
  unsigned int eax, ebx, ecx, edx;
#if defined(_MSC_VER)
  int regs[4];
  __cpuid(regs, 1);
  eax = regs[0]; ebx = regs[1]; ecx = regs[2]; edx = regs[3];
#elif defined(__GNUC__) || defined(__clang__)
  if (!__get_cpuid(1, &eax, &ebx, &ecx, &edx)) return false;
#endif
  const bool osxsave = (ecx & (1u << 27)) != 0u;
  const bool avx     = (ecx & (1u << 28)) != 0u;
  if (!(osxsave && avx)) return false;

  // Step 2: XCR0 must enable YMM state (bits 1 and 2)
  const std::uint64_t xcr0 = xgetbv_xcr0();
  const bool ymm_enabled = ((xcr0 & 0x6) == 0x6);
  if (!ymm_enabled) return false;

  // Step 3: CPUID.7.0:EBX bit 5 indicates AVX2
#if defined(_MSC_VER)
  int regs7[4];
  __cpuidex(regs7, 7, 0);
  ebx = static_cast<unsigned int>(regs7[1]);
#elif defined(__GNUC__) || defined(__clang__)
  unsigned int eax7, ecx7, edx7;
  if (!__get_cpuid_count(7, 0, &eax7, &ebx, &ecx7, &edx7)) return false;
#endif
  return (ebx & (1u << 5)) != 0u;
#endif
}

inline bool DetectNEON() {
#if defined(__aarch64__) || defined(_M_ARM64)
  // Advanced SIMD is mandatory in AArch64
  return true;
#else
  return false;
#endif
}

inline std::uint32_t GetCpuFeatureMask() {
#if defined(HYPERSTREAM_FORCE_SCALAR)
  return 0u;
#else
  std::uint32_t mask = 0u;
  if (DetectSSE2()) mask |= static_cast<std::uint32_t>(CpuFeature::SSE2);
  if (DetectAVX2()) mask |= static_cast<std::uint32_t>(CpuFeature::AVX2);
  if (DetectNEON()) mask |= static_cast<std::uint32_t>(CpuFeature::NEON);
  return mask;
#endif
}

} // namespace backend
} // namespace hyperstream

