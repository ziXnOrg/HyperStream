#include <gtest/gtest.h>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"

namespace {

using hyperstream::core::HyperVector;
using hyperstream::memory::CleanupMemory;
using hyperstream::memory::ClusterMemory;
using hyperstream::memory::PrototypeMemory;

TEST(PrototypeMemory, ClassifiesNearestNeighbor) {
  static constexpr std::size_t kDim = 64;
  static constexpr std::size_t kCap = 4;
  PrototypeMemory<kDim, kCap> memory;

  HyperVector<kDim, bool> hv_a;
  HyperVector<kDim, bool> hv_b;
  hv_a.Clear();
  hv_b.Clear();
  hv_a.SetBit(0, true);
  hv_a.SetBit(1, true);
  hv_b.SetBit(10, true);
  hv_b.SetBit(11, true);

  ASSERT_TRUE(memory.Learn(1, hv_a));
  ASSERT_TRUE(memory.Learn(2, hv_b));

  HyperVector<kDim, bool> query;
  query.Clear();
  query.SetBit(0, true);
  query.SetBit(1, true);
  query.SetBit(2, true);  // slightly noisy

  EXPECT_EQ(memory.Classify(query, 0), 1u);
}

TEST(PrototypeMemory, ClassifyReturnsDefaultLabelWhenEmpty) {
  static constexpr std::size_t kDim = 64;
  static constexpr std::size_t kCap = 4;
  PrototypeMemory<kDim, kCap> memory;

  // No Learn() calls; memory is empty
  ASSERT_EQ(memory.size(), 0u);

  HyperVector<kDim, bool> query;
  query.Clear();
  query.SetBit(3, true);  // arbitrary, should not matter for empty memory

  const std::uint64_t kDefault = 12345u;
  EXPECT_EQ(memory.Classify(query, kDefault), kDefault);
}


TEST(ClusterMemory, UpdateAndFinalizeReflectsMajority) {
  static constexpr std::size_t kDim = 32;
  static constexpr std::size_t kCap = 2;
  ClusterMemory<kDim, kCap> memory;

  HyperVector<kDim, bool> hv_1;
  HyperVector<kDim, bool> hv_2;
  hv_1.Clear();
  hv_2.Clear();
  for (std::size_t i = 0; i < 8; ++i) {
    hv_1.SetBit(i, true);
  }
  for (std::size_t i = 4; i < 12; ++i) {
    hv_2.SetBit(i, true);
  }

  ASSERT_TRUE(memory.Update(42, hv_1));
  ASSERT_TRUE(memory.Update(42, hv_2));

  HyperVector<kDim, bool> finalized;
  memory.Finalize(42, &finalized);
  for (std::size_t i = 0; i < 12; ++i) {
    const bool expected = (i < 12) ? true : false;
    EXPECT_EQ(finalized.GetBit(i), expected) << "bit index " << i;
  }
}

TEST(ClusterMemory, DecayReducesCounts) {
  static constexpr std::size_t kDim = 16;
  ClusterMemory<kDim, 1> memory;

  HyperVector<kDim, bool> hv;
  hv.Clear();
  hv.SetBit(0, true);
  hv.SetBit(1, true);

  ASSERT_TRUE(memory.Update(7, hv));
  memory.ApplyDecay(0.5f);

  HyperVector<kDim, bool> finalized;
  memory.Finalize(7, &finalized);
  EXPECT_TRUE(finalized.GetBit(0));
  EXPECT_TRUE(finalized.GetBit(1));
}

TEST(CleanupMemory, RestoreReturnsNearestStoredHV) {
  static constexpr std::size_t kDim = 64;
  CleanupMemory<kDim, 3> cleanup;

  HyperVector<kDim, bool> hv_clean;
  HyperVector<kDim, bool> hv_alt;
  hv_clean.Clear();
  hv_alt.Clear();
  for (std::size_t i = 0; i < 16; ++i) {
    hv_clean.SetBit(i, true);
  }
  for (std::size_t i = 32; i < 48; ++i) {
    hv_alt.SetBit(i, true);
  }
  ASSERT_TRUE(cleanup.Insert(hv_clean));
  ASSERT_TRUE(cleanup.Insert(hv_alt));

  HyperVector<kDim, bool> noisy = hv_clean;
  noisy.SetBit(20, true);
  noisy.SetBit(21, true);

  HyperVector<kDim, bool> fallback;
  fallback.Clear();

  const HyperVector<kDim, bool> restored = cleanup.Restore(noisy, fallback);
  for (std::size_t i = 0; i < hv_clean.Words().size(); ++i) {
    EXPECT_EQ(restored.Words()[i], hv_clean.Words()[i]) << "word index " << i;
  }
}

}  // namespace
