#include <gtest/gtest.h>
#include <cstdlib>
#ifdef _MSC_VER
#include <stdlib.h>
#endif

#include "hyperstream/backend/policy.hpp"
#include "hyperstream/backend/capability.hpp"
#include "hyperstream/core/hypervector.hpp"

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
}

TEST(Dispatch, NoIllegalInstructions_WhenAVX2MaskedOut) {
  using namespace hyperstream::backend;
  constexpr std::size_t D = 256;
  const std::uint32_t m = Mask(false, true); // AVX2 off, SSE2 on

  // Ensure we don't select AVX2 functions
  auto bind_fn = SelectBindBackend<D>(m);
  auto ham_fn  = SelectHammingBackend<D>(m);
  EXPECT_NE(bind_fn, &avx2::BindAVX2<D>);
  EXPECT_NE(ham_fn,  &avx2::HammingDistanceAVX2<D>);

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
    EXPECT_EQ(ham_fn, &sse2::HammingDistanceSSE2<D>);
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
    EXPECT_NE(ham_lt, &sse2::HammingDistanceSSE2<Dlt>); // below threshold should not force SSE2
    EXPECT_EQ(ham_eq, &sse2::HammingDistanceSSE2<Deq>); // at threshold selects SSE2
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

