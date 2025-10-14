#include <gtest/gtest.h>
#include <random>

#include "hyperstream/backend/cpu_backend.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

namespace {

using hyperstream::backend::Bind;
using hyperstream::backend::CpuBackend;
using hyperstream::backend::HammingDistance;
using hyperstream::core::HyperVector;

TEST(CpuBackend, DetectsFeatures) {
  const auto& backend = CpuBackend::Instance();
  const auto& features = backend.Features();

  // At minimum, scalar path should always work.
  // On x86-64, we expect SSE2 to be present.
#if defined(__x86_64__) || defined(_M_X64)
  EXPECT_TRUE(features.sse2);
#endif

  // Log detected features for manual inspection.
  std::cout << "CPU Features - AVX2: " << features.avx2 << ", SSE2: " << features.sse2 << std::endl;
}

TEST(CpuBackend, BindProducesCorrectXor) {
  static constexpr std::size_t kDim = 128;

  HyperVector<kDim, bool> a, b, result;
  a.Clear();
  b.Clear();

  // Set some bits in a and b.
  a.SetBit(0, true);
  a.SetBit(1, true);
  a.SetBit(10, true);

  b.SetBit(1, true);
  b.SetBit(5, true);
  b.SetBit(10, true);

  hyperstream::backend::Bind<kDim>(a, b, &result);

  // XOR: a[0]=1 b[0]=0 => result[0]=1
  EXPECT_TRUE(result.GetBit(0));
  // XOR: a[1]=1 b[1]=1 => result[1]=0
  EXPECT_FALSE(result.GetBit(1));
  // XOR: a[5]=0 b[5]=1 => result[5]=1
  EXPECT_TRUE(result.GetBit(5));
  // XOR: a[10]=1 b[10]=1 => result[10]=0
  EXPECT_FALSE(result.GetBit(10));
}

TEST(CpuBackend, HammingDistanceMatchesManualCount) {
  static constexpr std::size_t kDim = 256;

  HyperVector<kDim, bool> a, b;
  a.Clear();
  b.Clear();

  // Set 10 bits in a, 8 bits in b, with 5 overlapping.
  for (std::size_t i = 0; i < 10; ++i) {
    a.SetBit(i, true);
  }
  for (std::size_t i = 5; i < 13; ++i) {
    b.SetBit(i, true);
  }

  // Hamming distance = bits that differ.
  // a has bits [0..9], b has bits [5..12]
  // Overlap: [5..9] (5 bits)
  // a-only: [0..4] (5 bits)
  // b-only: [10..12] (3 bits)
  // Total differing: 5 + 3 = 8
  const std::size_t dist = hyperstream::backend::HammingDistance<kDim>(a, b);
  EXPECT_EQ(dist, 8u);
}

TEST(CpuBackend, BindIsInvertible) {
  static constexpr std::size_t kDim = 512;

  HyperVector<kDim, bool> original, key, bound, recovered;
  original.Clear();
  key.Clear();

  // Set some pattern in original.
  for (std::size_t i = 0; i < kDim; i += 7) {
    original.SetBit(i, true);
  }

  // Set key pattern.
  for (std::size_t i = 0; i < kDim; i += 13) {
    key.SetBit(i, true);
  }

  // Bind original with key.
  hyperstream::backend::Bind<kDim>(original, key, &bound);

  // Unbind (XOR again with same key).
  hyperstream::backend::Bind<kDim>(bound, key, &recovered);

  // Should recover original.
  for (std::size_t i = 0; i < original.Words().size(); ++i) {
    EXPECT_EQ(recovered.Words()[i], original.Words()[i]) << "word index " << i;
  }
}

TEST(CpuBackend, HammingDistanceIsSymmetric) {
  static constexpr std::size_t kDim = 1024;

  HyperVector<kDim, bool> a, b;
  a.Clear();
  b.Clear();

  a.SetBit(42, true);
  a.SetBit(100, true);
  b.SetBit(42, true);
  b.SetBit(200, true);

  const std::size_t dist_ab = hyperstream::backend::HammingDistance<kDim>(a, b);
  const std::size_t dist_ba = hyperstream::backend::HammingDistance<kDim>(b, a);

  EXPECT_EQ(dist_ab, dist_ba);
}

// Helper to compare backend vs core agreement for bind and hamming at a given dimension.
template <std::size_t Dim>
static void CheckBackendCoreAgreement() {
  using hyperstream::core::HyperVector;

  HyperVector<Dim, bool> a, b, core_out, backend_out;
  a.Clear();
  b.Clear();

  // Deterministic bit patterns.
  for (std::size_t i = 0; i < Dim; ++i) {
    const bool abit = ((i * 73 + 11) % 97) < 37;
    const bool bbit = ((i * 29 + 7) % 101) < 43;
    if (abit) a.SetBit(i, true);
    if (bbit) b.SetBit(i, true);
  }

  // Bind: backend vs core
  hyperstream::backend::Bind<Dim>(a, b, &backend_out);
  hyperstream::core::Bind<Dim>(a, b, &core_out);

  ASSERT_EQ(core_out.Words().size(), backend_out.Words().size());
  for (std::size_t w = 0; w < core_out.Words().size(); ++w) {
    EXPECT_EQ(core_out.Words()[w], backend_out.Words()[w]) << "word index " << w;
  }

  // Hamming distance: backend vs core
  const std::size_t dist_core = hyperstream::core::HammingDistance<Dim>(a, b);
  const std::size_t dist_backend = hyperstream::backend::HammingDistance<Dim>(a, b);
  EXPECT_EQ(dist_core, dist_backend);
}

TEST(CpuBackend, AgreesWithCoreOnAwkwardDims) {
  CheckBackendCoreAgreement<1>();
  CheckBackendCoreAgreement<63>();
  CheckBackendCoreAgreement<64>();
  CheckBackendCoreAgreement<65>();
  CheckBackendCoreAgreement<100>();
  CheckBackendCoreAgreement<127>();
  CheckBackendCoreAgreement<128>();
  CheckBackendCoreAgreement<129>();
  CheckBackendCoreAgreement<10000>();
}

TEST(CpuBackend, AgreesWithCoreOnRandomVectors_FixedSeed) {
  // For several dims, sample random pairs with fixed seed and verify backend==core
  auto run_for_dim = [](auto dim_tag) {
    constexpr std::size_t D = decltype(dim_tag)::value;
    std::mt19937 gen(42);
    std::bernoulli_distribution dist(0.5);
    for (int iter = 0; iter < 40; ++iter) {
      HyperVector<D, bool> a, b, out_core, out_backend;
      a.Clear();
      b.Clear();
      for (std::size_t i = 0; i < D; ++i) {
        if (dist(gen)) a.SetBit(i, true);
        if (dist(gen)) b.SetBit(i, true);
      }
      // Bind
      hyperstream::core::Bind<D>(a, b, &out_core);
      hyperstream::backend::Bind<D>(a, b, &out_backend);
      ASSERT_EQ(out_core.Words().size(), out_backend.Words().size());
      for (std::size_t w = 0; w < out_core.Words().size(); ++w) {
        EXPECT_EQ(out_core.Words()[w], out_backend.Words()[w]) << "word index " << w;
      }
      // Hamming
      const auto h_core = hyperstream::core::HammingDistance<D>(a, b);
      const auto h_back = hyperstream::backend::HammingDistance<D>(a, b);
      EXPECT_EQ(h_core, h_back);
    }
  };
  run_for_dim(std::integral_constant<std::size_t, 64>{});
  run_for_dim(std::integral_constant<std::size_t, 256>{});
  run_for_dim(std::integral_constant<std::size_t, 1000>{});
}

}  // namespace
