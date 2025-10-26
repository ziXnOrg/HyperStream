#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/io/serialization.hpp"
#include "hyperstream/memory/associative.hpp"

#ifndef HYPERSTREAM_TESTS_DIR
#define HYPERSTREAM_TESTS_DIR "tests"
#endif

namespace {

using hyperstream::core::HyperVector;
using hyperstream::io::LoadCluster;
using hyperstream::io::LoadPrototype;
using hyperstream::io::SaveCluster;
using hyperstream::io::SavePrototype;
using hyperstream::memory::ClusterMemory;
using hyperstream::memory::PrototypeMemory;

TEST(Serialization, Prototype_RoundTrip) {
  static constexpr std::size_t D = 128;
  static constexpr std::size_t C = 4;
  PrototypeMemory<D, C> mem;

  // Populate with 3 entries
  for (std::size_t i = 0; i < 3; ++i) {
    HyperVector<D, bool> hv;
    hv.Clear();
    for (std::size_t b = i; b < D; b += (i + 1)) hv.SetBit(b, true);
    ASSERT_TRUE(mem.Learn(static_cast<std::uint64_t>(i + 1), hv));
  }

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss, mem));

  PrototypeMemory<D, C> loaded;
  ASSERT_TRUE(LoadPrototype(ss, &loaded));
  ASSERT_EQ(loaded.Size(), mem.Size());

  const auto* a = mem.Data();
  const auto* b = loaded.Data();
  for (std::size_t i = 0; i < mem.Size(); ++i) {
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
  HyperVector<D, bool> hv;
  hv.Clear();
  hv.SetBit(1, true);
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
    HyperVector<D, bool> a;
    a.Clear();
    for (std::size_t i = 0; i < D; i += 3) a.SetBit(i, true);
    ASSERT_TRUE(mem.Update(1001, a));
  }
  for (int rep = 0; rep < 3; ++rep) {
    HyperVector<D, bool> b;
    b.Clear();
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
  HyperVector<D1, bool> a;
  a.Clear();
  a.SetBit(0, true);
  ASSERT_TRUE(mem.Update(7, a));

  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss, mem));
  ClusterMemory<D2, C> bad;
  EXPECT_FALSE(LoadCluster(ss, &bad));
}

}  // namespace

TEST(Serialization_V11, Prototype_RoundTrip_WithTrailerCRC) {
  static constexpr std::size_t D = 128;
  static constexpr std::size_t C = 4;
  PrototypeMemory<D, C> mem;
  // 3 entries with simple patterns
  for (std::size_t i = 0; i < 3; ++i) {
    HyperVector<D, bool> hv;
    hv.Clear();
    for (std::size_t b = i; b < D; b += (i + 1)) hv.SetBit(b, true);
    ASSERT_TRUE(mem.Learn(static_cast<std::uint64_t>(i + 11), hv));
  }
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss, mem));
  const std::string blob = ss.str();
  // Expect trailer tag "HSX1" at end-8
  ASSERT_GE(blob.size(), 8u);
  const char* end = blob.data() + blob.size();
  EXPECT_EQ(end[-8], 'H');
  EXPECT_EQ(end[-7], 'S');
  EXPECT_EQ(end[-6], 'X');
  EXPECT_EQ(end[-5], '1');
  // Load back and compare
  std::stringstream in(std::ios::in | std::ios::out | std::ios::binary);
  in.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  PrototypeMemory<D, C> loaded;
  ASSERT_TRUE(LoadPrototype(in, &loaded));
  ASSERT_EQ(loaded.Size(), mem.Size());
  const auto* a = mem.Data();
  const auto* b = loaded.Data();
  for (std::size_t i = 0; i < mem.Size(); ++i) {
    EXPECT_EQ(a[i].label, b[i].label);
    for (std::size_t w = 0; w < a[i].hv.Words().size(); ++w) {
      EXPECT_EQ(a[i].hv.Words()[w], b[i].hv.Words()[w]);
    }
  }
}

TEST(Serialization_V11, Prototype_CorruptionDetected) {
  static constexpr std::size_t D = 64;
  static constexpr std::size_t C = 2;
  PrototypeMemory<D, C> mem;
  // Single entry
  {
    HyperVector<D, bool> hv;
    hv.Clear();
    for (std::size_t b = 0; b < D; b += 2) hv.SetBit(b, true);
    ASSERT_TRUE(mem.Learn(0xABCDEFu, hv));
  }
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss, mem));
  std::string blob = ss.str();
  // Flip a payload byte (after header) but not the trailer
  struct LocalHeader {
    char magic[5];
    unsigned char kind;
    std::uint64_t dim, cap, size;
  };
  ASSERT_GE(blob.size(), sizeof(LocalHeader) + 16);
  blob[sizeof(LocalHeader) + 1] ^= 0x01;
  std::stringstream bad(std::ios::in | std::ios::out | std::ios::binary);
  bad.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  PrototypeMemory<D, C> out;
  EXPECT_FALSE(LoadPrototype(bad, &out));
}

TEST(Serialization_BackCompat, Load_V1_Goldens) {
  // Load existing v1 goldens (no trailer) and verify basic invariants
  // Prototype d128/c4
  {
    static constexpr std::size_t D = 128;
    static constexpr std::size_t C = 4;
    const std::string path_proto =
        std::string(HYPERSTREAM_TESTS_DIR) + "/golden/hser1/prototype_d128_c4.hser1";
    std::ifstream f(path_proto, std::ios::binary);
    ASSERT_TRUE(static_cast<bool>(f));
    PrototypeMemory<D, C> mem;
    ASSERT_TRUE(LoadPrototype(f, &mem));
    ASSERT_LE(mem.Size(), C);
  }
  // Cluster d96/c3
  {
    static constexpr std::size_t D = 96;
    static constexpr std::size_t C = 3;
    const std::string path_cluster =
        std::string(HYPERSTREAM_TESTS_DIR) + "/golden/hser1/cluster_d96_c3.hser1";
    std::ifstream f(path_cluster, std::ios::binary);
    ASSERT_TRUE(static_cast<bool>(f));
    ClusterMemory<D, C> mem;
    ASSERT_TRUE(LoadCluster(f, &mem));
    ASSERT_LE(mem.Size(), C);
  }
}

TEST(Serialization_V11, Cluster_CorruptionDetected) {
  static constexpr std::size_t D = 96;
  static constexpr std::size_t C = 3;
  ClusterMemory<D, C> mem;
  // Seed a few clusters via Learn path
  {
    HyperVector<D, bool> a;
    a.Clear();
    for (std::size_t i = 0; i < D; i += 3) a.SetBit(i, true);
    HyperVector<D, bool> b;
    b.Clear();
    for (std::size_t i = 1; i < D; i += 3) b.SetBit(i, true);
    ASSERT_TRUE(mem.Update(10, a));
    ASSERT_TRUE(mem.Update(20, b));
  }
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss, mem));
  std::string blob = ss.str();
  struct LocalHeader {
    char magic[5];
    unsigned char kind;
    std::uint64_t dim, cap, size;
  };
  ASSERT_GE(blob.size(), sizeof(LocalHeader) + 16);
  blob[sizeof(LocalHeader) + 2] ^= 0x01;  // flip a payload byte
  std::stringstream bad(std::ios::in | std::ios::out | std::ios::binary);
  bad.write(blob.data(), static_cast<std::streamsize>(blob.size()));
  ClusterMemory<D, C> out;
  EXPECT_FALSE(LoadCluster(bad, &out));
}
