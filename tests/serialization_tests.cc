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

TEST(Serialization, Prototype_RoundTrip) {
  static constexpr std::size_t D = 128;
  static constexpr std::size_t C = 4;
  PrototypeMemory<D, C> mem;

  // Populate with 3 entries
  for (std::size_t i = 0; i < 3; ++i) {
    HyperVector<D, bool> hv; hv.Clear();
    for (std::size_t b = i; b < D; b += (i+1)) hv.SetBit(b, true);
    ASSERT_TRUE(mem.Learn(static_cast<std::uint64_t>(i + 1), hv));
  }

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss, mem));

  PrototypeMemory<D, C> loaded;
  ASSERT_TRUE(LoadPrototype(ss, &loaded));
  ASSERT_EQ(loaded.size(), mem.size());

  const auto* a = mem.data();
  const auto* b = loaded.data();
  for (std::size_t i = 0; i < mem.size(); ++i) {
    EXPECT_EQ(a[i].label, b[i].label);
    for (std::size_t w = 0; w < a[i].hv.Words().size(); ++w) {
      EXPECT_EQ(a[i].hv.Words()[w], b[i].hv.Words()[w]) << "entry " << i << ", word " << w;
    }
  }
}

TEST(Serialization, Prototype_BadMagicFails) {
  static constexpr std::size_t D = 64;
  static constexpr std::size_t C = 2;
  PrototypeMemory<D, C> mem;
  HyperVector<D, bool> hv; hv.Clear(); hv.SetBit(1, true);
  ASSERT_TRUE(mem.Learn(42, hv));

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss, mem));
  // Corrupt first byte
  std::string s = ss.str();
  s[0] ^= 0xFF;
  std::stringstream bad(std::ios::in | std::ios::out | std::ios::binary);
  bad.write(s.data(), static_cast<std::streamsize>(s.size()));
  PrototypeMemory<D, C> out;
  EXPECT_FALSE(LoadPrototype(bad, &out));
}

TEST(Serialization, Cluster_RoundTrip) {
  static constexpr std::size_t D = 96;
  static constexpr std::size_t C = 3;
  ClusterMemory<D, C> mem;

  // Build two clusters with different patterns.
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

  HyperVector<D, bool> a_fin, b_fin;
  mem.Finalize(1001, &a_fin);
  mem.Finalize(2002, &b_fin);

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss, mem));

  ClusterMemory<D, C> loaded;
  ASSERT_TRUE(LoadCluster(ss, &loaded));

  HyperVector<D, bool> a_fin2, b_fin2;
  loaded.Finalize(1001, &a_fin2);
  loaded.Finalize(2002, &b_fin2);

  for (std::size_t w = 0; w < a_fin.Words().size(); ++w) {
    EXPECT_EQ(a_fin.Words()[w], a_fin2.Words()[w]) << "A word " << w;
    EXPECT_EQ(b_fin.Words()[w], b_fin2.Words()[w]) << "B word " << w;
  }
}

TEST(Serialization, Cluster_DimMismatchFails) {
  static constexpr std::size_t D1 = 64;
  static constexpr std::size_t D2 = 128;
  static constexpr std::size_t C = 2;
  ClusterMemory<D1, C> mem;
  HyperVector<D1, bool> a; a.Clear(); a.SetBit(0,true);
  ASSERT_TRUE(mem.Update(7, a));

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss, mem));
  ClusterMemory<D2, C> bad;
  EXPECT_FALSE(LoadCluster(ss, &bad));
}

}  // namespace

