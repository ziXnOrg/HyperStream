#include <gtest/gtest.h>
#include <random>
#include <array>

#include "hyperstream/backend/policy.hpp"
#include "hyperstream/backend/capability.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define HS_X86_ARCH 1
#else
#define HS_X86_ARCH 0
#endif

using hyperstream::core::HyperVector;

namespace {

using namespace hyperstream::backend;

static std::uint32_t MaskNone() { return 0u; }
static std::uint32_t MaskSSE2() { return static_cast<std::uint32_t>(CpuFeature::SSE2); }
static std::uint32_t MaskAVX2() { return static_cast<std::uint32_t>(CpuFeature::AVX2); }
static std::uint32_t MaskBoth() { return MaskSSE2() | MaskAVX2(); }

// Deterministic bit fill for a and b
template <std::size_t D>
static void FillRandom(HyperVector<D, bool>& hv, std::mt19937& rng) {
  std::uniform_int_distribution<int> bit(0, 1);
  for (std::size_t i = 0; i < D; ++i) hv.SetBit(i, bit(rng) != 0);
}

// Execute invariants and small executions for a single dimension D
template <std::size_t D>
static void CheckOneDim() {
  const std::size_t thr = GetHammingThreshold();
  const std::array<std::uint32_t, 4> masks = {MaskNone(), MaskSSE2(), MaskAVX2(), MaskBoth()};

  for (std::uint32_t m : masks) {
    auto bind_fn = SelectBindBackend<D>(m);
    auto ham_fn  = SelectHammingBackend<D>(m);

    // Non-null function pointers
    ASSERT_NE(bind_fn, nullptr);
    ASSERT_NE(ham_fn,  nullptr);

    const bool has_sse2 = HasFeature(m, CpuFeature::SSE2);
    const bool has_avx2 = HasFeature(m, CpuFeature::AVX2);

#if HS_X86_ARCH
    // Masking invariants
    if (!has_avx2) {
      EXPECT_NE(bind_fn, &avx2::BindAVX2<D>);
      EXPECT_NE(ham_fn,  &avx2::HammingDistanceAVX2<D>);
    }
    if (!has_sse2) {
      EXPECT_NE(bind_fn, &sse2::BindSSE2<D>);
      EXPECT_NE(ham_fn,  &sse2::HammingDistanceSSE2<D>);
    }

    // Scalar fallback when no features
    if (!has_sse2 && !has_avx2) {
      EXPECT_EQ(bind_fn, &hyperstream::core::Bind<D>);
      EXPECT_EQ(ham_fn,  &hyperstream::core::HammingDistance<D>);
    }

    // Threshold heuristic for Hamming when both SSE2+AVX2 available
    if (has_sse2 && has_avx2) {
      if (D >= thr) {
        EXPECT_EQ(ham_fn, &sse2::HammingDistanceSSE2<D>);
      } else {
        // Below threshold, should not prefer SSE2
        EXPECT_NE(ham_fn, &sse2::HammingDistanceSSE2<D>);
      }
    }

    // Bind always prefers AVX2 over SSE2 when both available
    if (has_sse2 && has_avx2) {
      EXPECT_NE(bind_fn, &sse2::BindSSE2<D>);
      EXPECT_NE(bind_fn, &hyperstream::core::Bind<D>);
    }
#else
    // Non-x86: all selections should be scalar regardless of mask
    EXPECT_EQ(bind_fn, &hyperstream::core::Bind<D>);
    EXPECT_EQ(ham_fn,  &hyperstream::core::HammingDistance<D>);
#endif

    // Execute selected backends to ensure no illegal instruction and correct results
    std::mt19937 rng(42);
    HyperVector<D, bool> a, b, out, out_ref;
    a.Clear(); b.Clear(); out.Clear(); out_ref.Clear();
    FillRandom(a, rng); FillRandom(b, rng);

    // Bind correctness
    bind_fn(a, b, &out);
    hyperstream::core::Bind<D>(a, b, &out_ref);
    EXPECT_EQ(out.Words(), out_ref.Words());

    // Hamming correctness
    const auto d_sel = ham_fn(a, b);
    const auto d_ref = hyperstream::core::HammingDistance<D>(a, b);
    EXPECT_EQ(d_sel, d_ref);
  }
}

}  // namespace

TEST(DispatchProperty, InvariantsAcrossDimsAndMasks) {
  // Representative dims: powers of two, awkward sizes, and threshold neighbors
  // Note: compile-time sizes; invoke as templated calls
  CheckOneDim<64>();
  CheckOneDim<127>();
  CheckOneDim<128>();
  CheckOneDim<256>();
  CheckOneDim<1024>();
  CheckOneDim<8192>();
  CheckOneDim<16383>();
  CheckOneDim<16384>();
  CheckOneDim<16385>();
  CheckOneDim<32768>();
  CheckOneDim<65536>();
}

// Task 4 scaffolding: future Bind dimension-based heuristic tests
TEST(DispatchProperty, DISABLED_BindThreshold_TBD) {
  // TODO: If future benchmarking introduces a Bind size-based heuristic similar to
  // Hamming, add cases here to validate AVX2↔SSE2 selection at boundary dimensions.
  SUCCEED();
}

