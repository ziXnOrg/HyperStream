#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <vector>

// Force v1 writer for golden tests to match committed v1 fixtures
#define HYPERSTREAM_HSER1_WRITE_V1

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/io/serialization.hpp"
#include "hyperstream/memory/associative.hpp"

namespace {

using hyperstream::core::HyperVector;
using hyperstream::io::LoadCluster;
using hyperstream::io::LoadPrototype;
using hyperstream::io::SaveCluster;
using hyperstream::io::SavePrototype;
using hyperstream::memory::ClusterMemory;
using hyperstream::memory::PrototypeMemory;

static std::vector<std::uint8_t> StreamBytes(std::stringstream& ss) {
  const std::string s = ss.str();
  return std::vector<std::uint8_t>(s.begin(), s.end());
}

// We treat these tests as golden scaffolds:
// 1) Construct deterministic memories
// 2) Serialize to bytes
// 3) Load into fresh instances
// 4) Re-serialize and assert byte-for-byte idempotence
// This establishes a stable, byte-exact format foundation; static fixtures can be added
// subsequently.

TEST(GoldenSerialization, Prototype_ByteExactIdempotent) {
  static constexpr std::size_t D = 128;
  static constexpr std::size_t C = 4;
  PrototypeMemory<D, C> mem;

  for (std::size_t i = 0; i < 3; ++i) {
    HyperVector<D, bool> hv;
    hv.Clear();
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

// Helpers for file-based golden fixtures
static std::string TestsDir() {
#ifdef HYPERSTREAM_TESTS_DIR
  return std::string(HYPERSTREAM_TESTS_DIR);
#else
  return std::string("tests");
#endif
}

static std::string FixturePath(const std::string& name) {
  return TestsDir() + "/golden/hser1/" + name;
}

static std::vector<std::uint8_t> ReadFileBytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::vector<std::uint8_t> data;
  if (!f.good()) return data;
  f.seekg(0, std::ios::end);
  const std::streampos end = f.tellg();
  f.seekg(0, std::ios::beg);
  data.resize(static_cast<std::size_t>(end));
  if (!data.empty()) {
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
  }
  return data;
}

static void AssertBytesEqual(const std::vector<std::uint8_t>& a,
                             const std::vector<std::uint8_t>& b) {
  ASSERT_EQ(a.size(), b.size()) << "size mismatch";
  for (std::size_t i = 0; i < a.size(); ++i) {
    ASSERT_EQ(a[i], b[i]) << "byte index " << i;
  }
}

// Fixture-based tests: load committed bytes, deserialize, re-serialize, compare to fixture
TEST(GoldenSerialization, Prototype_MatchesCommittedFixture_D96_C3) {
  const auto bytes = ReadFileBytes(FixturePath("prototype_d96_c3.hser1"));
  ASSERT_FALSE(bytes.empty());
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));

  PrototypeMemory<96, 3> mem;
  ASSERT_TRUE(LoadPrototype(ss, &mem));
  ASSERT_EQ(mem.Size(), 2u);

  // Re-serialize and compare to committed fixture bytes
  std::stringstream ss2(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss2, mem));
  AssertBytesEqual(bytes, StreamBytes(ss2));

  // Semantic spot-check: query exactly one of the stored patterns
  HyperVector<96, bool> q;
  q.Clear();
  for (std::size_t i = 0; i < 96; i += 3) q.SetBit(i, true);
  EXPECT_EQ(mem.Classify(q, /*default=*/0), 1001u);
}

TEST(GoldenSerialization, Prototype_MatchesCommittedFixture_D128_C4) {
  const auto bytes = ReadFileBytes(FixturePath("prototype_d128_c4.hser1"));
  ASSERT_FALSE(bytes.empty());
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));

  PrototypeMemory<128, 4> mem;
  ASSERT_TRUE(LoadPrototype(ss, &mem));
  ASSERT_EQ(mem.Size(), 3u);

  std::stringstream ss2(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SavePrototype(ss2, mem));
  AssertBytesEqual(bytes, StreamBytes(ss2));

  HyperVector<128, bool> q;
  q.Clear();
  for (std::size_t b = 1; b < 128; b += 4) q.SetBit(b, true);
  EXPECT_EQ(mem.Classify(q, /*default=*/0), 2002u);
}

TEST(GoldenSerialization, Cluster_MatchesCommittedFixture_D96_C3) {
  const auto bytes = ReadFileBytes(FixturePath("cluster_d96_c3.hser1"));
  ASSERT_FALSE(bytes.empty());
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));

  ClusterMemory<96, 3> mem;
  ASSERT_TRUE(LoadCluster(ss, &mem));
  ASSERT_EQ(mem.Size(), 2u);

  std::stringstream ss2(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss2, mem));
  AssertBytesEqual(bytes, StreamBytes(ss2));

  // Semantic: finalize cluster 1001 and expect it to align with majority of its updates
  HyperVector<96, bool> out;
  mem.Finalize(1001, &out);
  for (std::size_t i = 0; i < 96; i += 3) {
    EXPECT_TRUE(out.GetBit(i));
  }
}

TEST(GoldenSerialization, Cluster_MatchesCommittedFixture_D128_C4) {
  const auto bytes = ReadFileBytes(FixturePath("cluster_d128_c4.hser1"));
  ASSERT_FALSE(bytes.empty());
  std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
  ss.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));

  ClusterMemory<128, 4> mem;
  ASSERT_TRUE(LoadCluster(ss, &mem));
  ASSERT_EQ(mem.Size(), 2u);

  std::stringstream ss2(std::ios::in | std::ios::out | std::ios::binary);
  ASSERT_TRUE(SaveCluster(ss2, mem));
  AssertBytesEqual(bytes, StreamBytes(ss2));

  HyperVector<128, bool> out;
  mem.Finalize(2002, &out);
  for (std::size_t b = 1; b < 128; b += 4) {
    EXPECT_TRUE(out.GetBit(b));
  }
}

// Minimal SHA-256 implementation (compact, test-only)
namespace sha256_internal {
static inline uint32_t rotr(uint32_t x, uint32_t n) {
  return (x >> n) | (x << (32 - n));
}
static const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
}  // namespace sha256_internal

static std::array<std::uint8_t, 32> Sha256(const std::vector<std::uint8_t>& msg) {
  using namespace sha256_internal;
  uint64_t bitlen = static_cast<uint64_t>(msg.size()) * 8ull;
  std::vector<uint8_t> m = msg;
  m.push_back(0x80);
  while ((m.size() % 64) != 56) m.push_back(0x00);
  for (int i = 7; i >= 0; --i) m.push_back(static_cast<uint8_t>((bitlen >> (i * 8)) & 0xFF));
  uint32_t H[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                   0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
  for (size_t off = 0; off < m.size(); off += 64) {
    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
      size_t j = off + i * 4;
      w[i] = (m[j] << 24) | (m[j + 1] << 16) | (m[j + 2] << 8) | m[j + 3];
    }
    for (int i = 16; i < 64; ++i) {
      uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
    for (int i = 0; i < 64; ++i) {
      uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      uint32_t ch = (e & f) ^ (~e & g);
      uint32_t temp1 = h + S1 + ch + K[i] + w[i];
      uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      uint32_t temp2 = S0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + temp1;
      d = c;
      c = b;
      b = a;
      a = temp1 + temp2;
    }
    H[0] += a;
    H[1] += b;
    H[2] += c;
    H[3] += d;
    H[4] += e;
    H[5] += f;
    H[6] += g;
    H[7] += h;
  }
  std::array<std::uint8_t, 32> out{};
  for (int i = 0; i < 8; ++i) {
    out[i * 4 + 0] = static_cast<uint8_t>((H[i] >> 24) & 0xFF);
    out[i * 4 + 1] = static_cast<uint8_t>((H[i] >> 16) & 0xFF);
    out[i * 4 + 2] = static_cast<uint8_t>((H[i] >> 8) & 0xFF);
    out[i * 4 + 3] = static_cast<uint8_t>((H[i]) & 0xFF);
  }
  return out;
}

static std::string ToHex(const std::array<std::uint8_t, 32>& d) {
  static const char* k = "0123456789abcdef";
  std::string s;
  s.resize(64);
  for (int i = 0; i < 32; ++i) {
    s[i * 2] = k[(d[i] >> 4) & 0xF];
    s[i * 2 + 1] = k[d[i] & 0xF];
  }
  return s;
}

static std::string Sha256Hex(const std::vector<std::uint8_t>& v) {
  return ToHex(Sha256(v));
}

static std::string ReadTextFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static std::string FindShaInManifest(const std::string& manifest, const std::string& file) {
  const std::string key = std::string("\"file\": \"") + file + "\"";
  const auto pos = manifest.find(key);
  if (pos == std::string::npos) return std::string();
  const auto sha_key = manifest.find("\"sha256\": \"", pos);
  if (sha_key == std::string::npos) return std::string();
  const auto start = sha_key + std::string("\"sha256\": \"").size();
  const auto end = manifest.find("\"", start);
  if (end == std::string::npos) return std::string();
  return manifest.substr(start, end - start);
}

TEST(GoldenSerialization, Manifest_Sha256MatchesFiles) {
  const std::string manifest_path = FixturePath("manifest.json");
  const std::string manifest = ReadTextFile(manifest_path);
  const char* files[] = {"prototype_d96_c3.hser1", "prototype_d128_c4.hser1",
                         "cluster_d96_c3.hser1", "cluster_d128_c4.hser1"};
  for (const char* fname : files) {
    const auto bytes = ReadFileBytes(FixturePath(fname));
    ASSERT_FALSE(bytes.empty()) << fname;
    const std::string expect = FindShaInManifest(manifest, fname);
    ASSERT_FALSE(expect.empty()) << "sha missing for " << fname;
    const std::string got = Sha256Hex(bytes);
    EXPECT_EQ(got, expect) << fname;
  }
}

// Disabled local generator: writes canonical fixtures into tests/golden/hser1/
TEST(GoldenSerialization, DISABLED_GenerateHser1Fixtures) {
  // Prototype 96/3 with 2 entries
  {
    PrototypeMemory<96, 3> mem;
    HyperVector<96, bool> a;
    a.Clear();
    for (std::size_t i = 0; i < 96; i += 3) a.SetBit(i, true);
    ASSERT_TRUE(mem.Learn(1001, a));
    HyperVector<96, bool> b;
    b.Clear();
    for (std::size_t i = 1; i < 96; i += 4) b.SetBit(i, true);
    ASSERT_TRUE(mem.Learn(2002, b));
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(SavePrototype(ss, mem));
    const auto bytes = StreamBytes(ss);
    std::ofstream out(FixturePath("prototype_d96_c3.hser1"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  // Prototype 128/4 with 3 entries
  {
    PrototypeMemory<128, 4> mem;
    HyperVector<128, bool> a;
    a.Clear();
    for (std::size_t i = 0; i < 128; i += 3) a.SetBit(i, true);
    ASSERT_TRUE(mem.Learn(1001, a));
    HyperVector<128, bool> b;
    b.Clear();
    for (std::size_t i = 1; i < 128; i += 4) b.SetBit(i, true);
    ASSERT_TRUE(mem.Learn(2002, b));
    HyperVector<128, bool> c;
    c.Clear();
    for (std::size_t i = 2; i < 128; i += 5) c.SetBit(i, true);
    ASSERT_TRUE(mem.Learn(3003, c));
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(SavePrototype(ss, mem));
    const auto bytes = StreamBytes(ss);
    std::ofstream out(FixturePath("prototype_d128_c4.hser1"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  // Cluster 96/3 with deterministic updates
  {
    ClusterMemory<96, 3> mem;
    for (int rep = 0; rep < 5; ++rep) {
      HyperVector<96, bool> a;
      a.Clear();
      for (std::size_t i = 0; i < 96; i += 3) a.SetBit(i, true);
      ASSERT_TRUE(mem.Update(1001, a));
    }
    for (int rep = 0; rep < 3; ++rep) {
      HyperVector<96, bool> b;
      b.Clear();
      for (std::size_t i = 1; i < 96; i += 4) b.SetBit(i, true);
      ASSERT_TRUE(mem.Update(2002, b));
    }
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(SaveCluster(ss, mem));
    const auto bytes = StreamBytes(ss);
    std::ofstream out(FixturePath("cluster_d96_c3.hser1"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
  // Cluster 128/4 with deterministic updates
  {
    ClusterMemory<128, 4> mem;
    for (int rep = 0; rep < 4; ++rep) {
      HyperVector<128, bool> a;
      a.Clear();
      for (std::size_t i = 0; i < 128; i += 3) a.SetBit(i, true);
      ASSERT_TRUE(mem.Update(1001, a));
    }
    for (int rep = 0; rep < 6; ++rep) {
      HyperVector<128, bool> b;
      b.Clear();
      for (std::size_t i = 1; i < 128; i += 4) b.SetBit(i, true);
      ASSERT_TRUE(mem.Update(2002, b));
    }
    std::stringstream ss(std::ios::in | std::ios::out | std::ios::binary);
    ASSERT_TRUE(SaveCluster(ss, mem));
    const auto bytes = StreamBytes(ss);
    std::ofstream out(FixturePath("cluster_d128_c4.hser1"), std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }
}

}  // namespace
