#include <gtest/gtest.h>

#include "hyperstream/config.hpp"

TEST(Config, DefaultsAndProfiles) {
  using namespace hyperstream::config;
  // Basic sanity
  EXPECT_GE(kDefaultDimBits, static_cast<std::size_t>(8));
  EXPECT_GE(kDefaultCapacity, static_cast<std::size_t>(1));

#if defined(HYPERSTREAM_PROFILE_EMBEDDED)
  EXPECT_EQ(kDefaultDimBits, static_cast<std::size_t>(2048));
  EXPECT_EQ(kDefaultCapacity, static_cast<std::size_t>(16));
#else
  // Desktop/default profile
  EXPECT_EQ(kDefaultDimBits, static_cast<std::size_t>(10000));
  EXPECT_EQ(kDefaultCapacity, static_cast<std::size_t>(256));
#endif

  // Helper function checks
  EXPECT_TRUE(IsPowerOfTwo(1));
  EXPECT_TRUE(IsPowerOfTwo(16));
  EXPECT_FALSE(IsPowerOfTwo(18));
}

TEST(Config, FootprintHelpersAreCorrectForSmallDims) {
  using namespace hyperstream::config;
  // Dim=64: 1 word -> 8 bytes
  EXPECT_EQ(BinaryHyperVectorStorageBytes(64), static_cast<std::size_t>(8));
  // PrototypeMemory: 2 entries of (label 8 + hv 8) = 32
  EXPECT_EQ(PrototypeMemoryStorageBytes(64, 2), static_cast<std::size_t>(32));
  // ClusterMemory: labels 2*8 + counts 2*4 + sums 2*64*4 = 8*2 + 4*2 + 512 = 536
  EXPECT_EQ(ClusterMemoryStorageBytes(64, 2), static_cast<std::size_t>(536));
  // CleanupMemory: 2 * 8 = 16
  EXPECT_EQ(CleanupMemoryStorageBytes(64, 2), static_cast<std::size_t>(16));
}
