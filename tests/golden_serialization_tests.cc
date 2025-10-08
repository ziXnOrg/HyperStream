#include <gtest/gtest.h>
#include <sstream>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"
#include "hyperstream/io/serialization.hpp"

namespace {

using hyperstream::core::HyperVector;
using hyperstream::memory::PrototypeMemory;
using hyperstream::memory::ClusterMemory;
using hyperstream::io::SavePrototype;
using hyperstream::io::LoadPrototype;
using hyperstream::io::SaveCluster;
using hyperstream::io::LoadCluster;

static std::vector<std::uint8_t> StreamBytes(std::stringstream& ss) {
  const std::string s = ss.str();
  return std::vector<std::uint8_t>(s.begin(), s.end());
}

// We treat these tests as golden scaffolds:
// 1) Construct deterministic memories
// 2) Serialize to bytes
// 3) Load into fresh instances
// 4) Re-serialize and assert byte-for-byte idempotence
// This establishes a stable, byte-exact format foundation; static fixtures can be added subsequently.

TEST(GoldenSerialization, Prototype_ByteExactIdempotent) {
  static constexpr std::size_t D = 128;
  static constexpr std::size_t C = 4;
  PrototypeMemory<D, C> mem;

  for (std::size_t i = 0; i < 3; ++i) {
    HyperVector<D, bool> hv; hv.Clear();
    for (std::size_t b = i; b < D; b += (i + 1)) hv.SetBit(b, true);
    ASSERT_TRUE(mem.Learn(static_cast<std::uint64_t>(i + 1), hv));
  }

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss, mem));
  const auto bytes1 = StreamBytes(ss);

  PrototypeMemory<D, C> loaded;
  ASSERT_TRUE(LoadPrototype(ss, &loaded));

  std::stringstream ss2(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss2, loaded));
  const auto bytes2 = StreamBytes(ss2);

  ASSERT_EQ(bytes1.size(), bytes2.size());
  for (std::size_t i = 0; i < bytes1.size(); ++i) {
    EXPECT_EQ(bytes1[i], bytes2[i]) << "byte index " << i;
  }
}

TEST(GoldenSerialization, Cluster_ByteExactIdempotent) {
  static constexpr std::size_t D = 96;
  static constexpr std::size_t C = 3;
  ClusterMemory<D, C> mem;

  for (int rep = 0; rep < 5; ++rep) {
    HyperVector<D, bool> a; a.Clear();
    for (std::size_t i = 0; i < D; i += 3) a.SetBit(i, true);
    ASSERT_TRUE(mem.Update(1001, a));
  }
  for (int rep = 0; rep < 3; ++rep) {
    HyperVector<D, bool> b; b.Clear();
    for (std::size_t i = 1; i < D; i += 4) b.SetBit(i, true);
    ASSERT_TRUE(mem.Update(2002, b));
  }

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss, mem));
  const auto bytes1 = StreamBytes(ss);

  ClusterMemory<D, C> loaded;
  ASSERT_TRUE(LoadCluster(ss, &loaded));
  std::stringstream ss2(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss2, loaded));
  const auto bytes2 = StreamBytes(ss2);

  ASSERT_EQ(bytes1.size(), bytes2.size());
  for (std::size_t i = 0; i < bytes1.size(); ++i) {
    EXPECT_EQ(bytes1[i], bytes2[i]) << "byte index " << i;
  }
}

} // namespace

