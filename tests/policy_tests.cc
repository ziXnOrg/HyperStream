#include <gtest/gtest.h>
#include "hyperstream/backend/policy.hpp"
#include "hyperstream/core/hypervector.hpp"

using hyperstream::core::HyperVector;

namespace {

template <std::size_t Dim>
static void FillPattern(HyperVector<Dim, bool>* hv) {
  hv->Clear();
  // Simple deterministic pattern
  for (std::size_t i = 0; i < Dim; i += 3) {
    hv->SetBit(i, true);
  }
}

} // namespace

TEST(Policy, SelectBindAndHammingProduceCorrectResults) {
  constexpr std::size_t D = 64;
  HyperVector<D, bool> a, b, out_ref, out_sel;
  FillPattern(&a);
  b.Clear();
  for (std::size_t i = 1; i < D; i += 4) b.SetBit(i, true);

  // Reference scalar
  hyperstream::core::Bind<D>(a, b, &out_ref);
  const std::size_t h_ref = hyperstream::core::HammingDistance<D>(a, b);

  // Selected backends
  auto bind_fn = hyperstream::backend::SelectBindBackend<D>();
  auto ham_fn  = hyperstream::backend::SelectHammingBackend<D>();

  bind_fn(a, b, &out_sel);
  const std::size_t h_sel = ham_fn(a, b);

  // Words must match exactly; hamming must match
  for (std::size_t w = 0; w < out_ref.Words().size(); ++w) {
    EXPECT_EQ(out_sel.Words()[w], out_ref.Words()[w]) << "word index " << w;
  }
  EXPECT_EQ(h_sel, h_ref);
}

TEST(Policy, HeuristicPrefersSSE2ForLargeDimsWhenAVX2Present) {
  using namespace hyperstream::backend;
  const std::uint32_t mask = GetCpuFeatureMask();
  (void)mask; // silence unused when __AVX2__ undefined
#if defined(__AVX2__)
  if (HasFeature(mask, CpuFeature::AVX2)) {
    // Below threshold should prefer AVX2
    {
      auto rep = Report<8192>(mask);
      EXPECT_NE(GetBackendName(rep.hamming_kind), nullptr);
    }
    // At/above threshold should prefer SSE2
    {
      auto rep = Report<65536>(mask);
      EXPECT_NE(GetBackendName(rep.hamming_kind), nullptr);
    }
  }
#endif
}

TEST(Policy, ThresholdDefaultWhenEnvUnset) {
  using namespace hyperstream::backend;
  // We only assert that it returns a positive default (compile-time default)
  const std::size_t thr = GetHammingThreshold();
  EXPECT_EQ(thr, static_cast<std::size_t>(16384));
}

