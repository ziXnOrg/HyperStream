#include <gtest/gtest.h>

#include "hyperstream/backend/capability.hpp"

TEST(Capability, DetectionAndMaskAreSelfConsistent) {
  using namespace hyperstream::backend;

#if defined(HYPERSTREAM_FORCE_SCALAR)
  EXPECT_EQ(GetCpuFeatureMask(), 0u);
#else
  const bool sse2 = DetectSSE2();
  const bool avx2 = DetectAVX2();
  const std::uint32_t mask = GetCpuFeatureMask();

  // If AVX2 is detected, SSE2 should also be considered present on x86_64.
  if (avx2) {
    EXPECT_TRUE(HasFeature(mask, CpuFeature::AVX2));
  }
  if (sse2) {
    EXPECT_TRUE(HasFeature(mask, CpuFeature::SSE2));
  }

  // Round-trip property: features implied by mask match detection (best effort)
  EXPECT_EQ(HasFeature(mask, CpuFeature::AVX2), avx2);
#endif
}
