#include <gtest/gtest.h>

// Intentionally compile this TU with HYPERSTREAM_FORCE_SCALAR defined via target_compile_definitions
#include "hyperstream/backend/policy.hpp"
#include "hyperstream/core/hypervector.hpp"

TEST(ForcedScalar, SelectsCoreFunctionsRegardlessOfMaskOrDim) {
  using namespace hyperstream::backend;
  constexpr std::size_t D1 = 64;
  constexpr std::size_t D2 = 65536;

  const std::uint32_t masks[] = {
    0u,
    static_cast<std::uint32_t>(CpuFeature::SSE2),
    static_cast<std::uint32_t>(CpuFeature::AVX2),
    static_cast<std::uint32_t>(CpuFeature::SSE2) | static_cast<std::uint32_t>(CpuFeature::AVX2)
  };

  for (std::uint32_t m : masks) {
    auto bind1 = SelectBindBackend<D1>(m);
    auto ham1  = SelectHammingBackend<D1>(m);
    auto bind2 = SelectBindBackend<D2>(m);
    auto ham2  = SelectHammingBackend<D2>(m);

    EXPECT_EQ(bind1, &hyperstream::core::Bind<D1>);
    EXPECT_EQ(ham1,  &hyperstream::core::HammingDistance<D1>);
    EXPECT_EQ(bind2, &hyperstream::core::Bind<D2>);
    EXPECT_EQ(ham2,  &hyperstream::core::HammingDistance<D2>);
  }

  // Execute to ensure runtime doesn't trap and outputs are sane
  hyperstream::core::HyperVector<D1, bool> a, b, out;
  a.Clear(); b.Clear();
  a.SetBit(1, true); b.SetBit(2, true);
  SelectBindBackend<D1>(0u)(a, b, &out);
  auto dist = SelectHammingBackend<D1>(0u)(a, b);
  EXPECT_GE(dist, static_cast<std::size_t>(0));
}

