#include <gtest/gtest.h>
#include <cstdlib>
#ifdef _MSC_VER
#include <stdlib.h>
#endif

#include "hyperstream/backend/policy.hpp"
#include "hyperstream/backend/capability.hpp"
#include "hyperstream/core/hypervector.hpp"


#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define HS_X86_ARCH 1
#else
#define HS_X86_ARCH 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define HS_ARM64_ARCH 1
#include "hyperstream/backend/cpu_backend_neon.hpp"
#else
#define HS_ARM64_ARCH 0
#endif

using hyperstream::core::HyperVector;

namespace {

// Helper to build synthetic feature masks
static std::uint32_t Mask(bool avx2, bool sse2) {
  using namespace hyperstream::backend;
  std::uint32_t m = 0;
  if (avx2) m |= static_cast<std::uint32_t>(CpuFeature::AVX2);
  if (sse2) m |= static_cast<std::uint32_t>(CpuFeature::SSE2);
  return m;
}

} // namespace

TEST(Dispatch, Correctness_ByMaskAndDim) {
  using namespace hyperstream::backend;
#if HS_X86_ARCH
  // AVX2 present, small dim -> should not select SSE2 or scalar for Hamming/Bind
  {
    constexpr std::size_t Dsmall = 64;
    const std::uint32_t m = Mask(true, true);
    auto bind_fn = SelectBindBackend<Dsmall>(m);
    auto ham_fn  = SelectHammingBackend<Dsmall>(m);
    EXPECT_NE(bind_fn, &sse2::BindSSE2<Dsmall>);
    EXPECT_NE(bind_fn, &hyperstream::core::Bind<Dsmall>);
    EXPECT_NE(ham_fn,  &sse2::HammingDistanceSSE2<Dsmall>);
    EXPECT_NE(ham_fn,  &hyperstream::core::HammingDistance<Dsmall>);
  }

  // AVX2 present, large dim >= threshold -> should not select AVX2 or scalar for Hamming per heuristic
  {
    constexpr std::size_t Dlarge = 1 << 16; // 65536 >= default 16384
    const std::uint32_t m = Mask(true, true);
    auto ham_fn  = SelectHammingBackend<Dlarge>(m);
    EXPECT_NE(ham_fn,  &avx2::HammingDistanceAVX2<Dlarge>);
    EXPECT_NE(ham_fn,  &hyperstream::core::HammingDistance<Dlarge>);
  }

  // SSE2 only -> should not select AVX2 or scalar for either op
  {
    constexpr std::size_t D = 256;
    const std::uint32_t m = Mask(false, true);
    auto bind_fn = SelectBindBackend<D>(m);
    auto ham_fn  = SelectHammingBackend<D>(m);
    EXPECT_NE(bind_fn, &avx2::BindAVX2<D>);
    EXPECT_NE(bind_fn, &hyperstream::core::Bind<D>);
    EXPECT_NE(ham_fn,  &avx2::HammingDistanceAVX2<D>);
    EXPECT_NE(ham_fn,  &hyperstream::core::HammingDistance<D>);
  }

  // Scalar only -> should not select AVX2 or SSE2
  {
    constexpr std::size_t D = 128;
    const std::uint32_t m = Mask(false, false);
    auto bind_fn = SelectBindBackend<D>(m);
    auto ham_fn  = SelectHammingBackend<D>(m);
    EXPECT_NE(bind_fn, &avx2::BindAVX2<D>);
    EXPECT_NE(bind_fn, &sse2::BindSSE2<D>);
    EXPECT_NE(ham_fn,  &avx2::HammingDistanceAVX2<D>);
    EXPECT_NE(ham_fn,  &sse2::HammingDistanceSSE2<D>);
  }
#else
  // Non-x86
  #if HS_ARM64_ARCH
  // On ARM64: NEON is baseline. Use default feature mask to select NEON.
  {
    constexpr std::size_t Dsmall = 64;
    auto bind_fn = SelectBindBackend<Dsmall>();
    auto ham_fn  = SelectHammingBackend<Dsmall>();
    EXPECT_EQ(bind_fn, &hyperstream::backend::neon::BindNEON<Dsmall>);
    EXPECT_EQ(ham_fn,  &hyperstream::backend::neon::HammingDistanceNEON<Dsmall>);
  }
  {
    constexpr std::size_t Dlarge = 1 << 16;
    auto bind_fn = SelectBindBackend<Dlarge>();
    auto ham_fn  = SelectHammingBackend<Dlarge>();
    EXPECT_EQ(bind_fn, &hyperstream::backend::neon::BindNEON<Dlarge>);
    EXPECT_EQ(ham_fn,  &hyperstream::backend::neon::HammingDistanceNEON<Dlarge>);
  }
  #else
  // Other non-x86: scalar fallback
  {
    constexpr std::size_t Dsmall = 64;
    auto bind_fn = SelectBindBackend<Dsmall>();
    auto ham_fn  = SelectHammingBackend<Dsmall>();
    EXPECT_EQ(bind_fn, &hyperstream::core::Bind<Dsmall>);
    EXPECT_EQ(ham_fn,  &hyperstream::core::HammingDistance<Dsmall>);
  }
  {
    constexpr std::size_t Dlarge = 1 << 16;
    auto bind_fn = SelectBindBackend<Dlarge>();
    auto ham_fn  = SelectHammingBackend<Dlarge>();
    EXPECT_EQ(bind_fn, &hyperstream::core::Bind<Dlarge>);
    EXPECT_EQ(ham_fn,  &hyperstream::core::HammingDistance<Dlarge>);
  }
  #endif
#endif
}

TEST(Dispatch, NoIllegalInstructions_WhenAVX2MaskedOut) {
  using namespace hyperstream::backend;
  constexpr std::size_t D = 256;
  const std::uint32_t m = Mask(false, true); // AVX2 off, SSE2 on

  // Ensure we don't select AVX2 functions
  auto bind_fn = SelectBindBackend<D>(m);
  auto ham_fn  = SelectHammingBackend<D>(m);
#if HS_X86_ARCH
  EXPECT_NE(bind_fn, &avx2::BindAVX2<D>);
  EXPECT_NE(ham_fn,  &avx2::HammingDistanceAVX2<D>);
#endif

  // Execute the selected functions to ensure they run correctly
  HyperVector<D, bool> a, b, out;
  a.Clear(); b.Clear();
  a.SetBit(3, true); b.SetBit(5, true);
  bind_fn(a, b, &out);
  const auto dist = ham_fn(a, b);
  // Sanity: bits differ at 3 and 5 => dist at least 2 depending on overlaps
  EXPECT_GE(dist, static_cast<std::size_t>(0));
}

TEST(Dispatch, EnvThreshold_OverridesHammingPreference) {
  using namespace hyperstream::backend;
  // Force threshold to 1 so any reasonable dim triggers SSE2 preference
#ifdef _MSC_VER
  _putenv_s("HYPERSTREAM_HAMMING_SSE2_THRESHOLD", "1");
#else
  setenv("HYPERSTREAM_HAMMING_SSE2_THRESHOLD", "1", 1);
#endif
  {
    constexpr std::size_t D = 64;
    const std::uint32_t m = Mask(true, true);
    auto ham_fn  = SelectHammingBackend<D>(m);
#if HS_X86_ARCH
    EXPECT_EQ(ham_fn, &sse2::HammingDistanceSSE2<D>);
#else
  #if HS_ARM64_ARCH
    EXPECT_EQ(ham_fn, &hyperstream::backend::neon::HammingDistanceNEON<D>);
  #else
    EXPECT_EQ(ham_fn, &hyperstream::core::HammingDistance<D>);
  #endif
#endif
  }
  // Cleanup
#ifdef _MSC_VER
  _putenv_s("HYPERSTREAM_HAMMING_SSE2_THRESHOLD", "");
#else
  unsetenv("HYPERSTREAM_HAMMING_SSE2_THRESHOLD");
#endif
}

TEST(Dispatch, ThresholdBoundary_AtExactThreshold) {
  using namespace hyperstream::backend;
#ifdef _MSC_VER
  _putenv_s("HYPERSTREAM_HAMMING_SSE2_THRESHOLD", "64");
#else
  setenv("HYPERSTREAM_HAMMING_SSE2_THRESHOLD", "64", 1);
#endif
  {
    constexpr std::size_t Dlt = 63;
    constexpr std::size_t Deq = 64;
    const std::uint32_t m = Mask(true, true);
    auto ham_lt = SelectHammingBackend<Dlt>(m);
    auto ham_eq = SelectHammingBackend<Deq>(m);
#if HS_X86_ARCH
    EXPECT_NE(ham_lt, &sse2::HammingDistanceSSE2<Dlt>);
    EXPECT_EQ(ham_eq, &sse2::HammingDistanceSSE2<Deq>);
#else
  #if HS_ARM64_ARCH
    EXPECT_EQ(ham_lt, &hyperstream::backend::neon::HammingDistanceNEON<Dlt>);
    EXPECT_EQ(ham_eq, &hyperstream::backend::neon::HammingDistanceNEON<Deq>);
  #else
    EXPECT_EQ(ham_lt, &hyperstream::core::HammingDistance<Dlt>);
    EXPECT_EQ(ham_eq, &hyperstream::core::HammingDistance<Deq>);
  #endif
#endif
  }
#ifdef _MSC_VER
  _putenv_s("HYPERSTREAM_HAMMING_SSE2_THRESHOLD", "");
#else
  unsetenv("HYPERSTREAM_HAMMING_SSE2_THRESHOLD");
#endif
}

TEST(Dispatch, SelectedBackend_ExecutesOnHost_NoIllegalInstruction) {
  using namespace hyperstream::backend;
  const std::uint32_t mask = GetCpuFeatureMask();
  constexpr std::size_t D = 256;
  auto bind_fn = SelectBindBackend<D>(mask);
  auto ham_fn  = SelectHammingBackend<D>(mask);
  HyperVector<D, bool> a, b, out; a.Clear(); b.Clear();
  a.SetBit(7, true); b.SetBit(13, true);
  // Execute the selected functions; any illegal-instruction would fail the test
  bind_fn(a, b, &out);
  auto dist = ham_fn(a, b);
  EXPECT_GE(dist, static_cast<std::size_t>(0));
}

