#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <string_view>
#include <fstream>
#include <sstream>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/encoding/item_memory.hpp"
#include "hyperstream/encoding/symbol.hpp"
#include "hyperstream/encoding/numeric.hpp"

namespace {

static std::string TestsDir() {
#ifdef HYPERSTREAM_TESTS_DIR
  return std::string(HYPERSTREAM_TESTS_DIR);
#else
  return std::string("tests");
#endif
}

static std::string PlatformId() {
#if defined(_WIN32)
  return std::string("windows-msvc");
#elif defined(__APPLE__)
  return std::string("macos-clang");
#elif defined(__linux__)
  #ifdef __clang__
    return std::string("linux-clang");
  #else
    return std::string("linux-gcc");
  #endif
#else
  return std::string("unknown");
#endif
}

static std::string ReadTextFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

static std::string FindHashJson(const std::string& json,
                                const std::string& encoder,
                                int dim,
                                const std::string& platform) {
  const std::string ekey = std::string("\"encoder\":\"") + encoder + "\"";
  const std::string dkey = std::string("\"dim\":") + std::to_string(dim);
  const std::string pkey = std::string("\"platform\":\"") + platform + "\"";
  std::size_t pos = 0;
  while (true) {
    pos = json.find(ekey, pos);
    if (pos == std::string::npos) return std::string();
    // Try to ensure we are in the same object by requiring dim and platform after encoder.
    const auto dpos = json.find(dkey, pos);
    const auto ppos = json.find(pkey, pos);
    if (dpos == std::string::npos || ppos == std::string::npos) { pos += ekey.size(); continue; }
    const auto hpos = json.find("\"hash\":\"", ppos);
    if (hpos == std::string::npos) { pos += ekey.size(); continue; }
    const auto start = hpos + std::string("\"hash\":\"").size();
    const auto end = json.find("\"", start);
    if (end == std::string::npos) return std::string();
    return json.substr(start, end - start);
  }
}

static std::string Hex64(std::uint64_t x) {
  char buf[19]; // 0x + 16 hex + \0
  std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(x));
  return std::string(buf);
}

using hyperstream::core::HyperVector;
using hyperstream::encoding::ItemMemory;
using hyperstream::encoding::SymbolEncoder;
using hyperstream::encoding::ThermometerEncoder;
using hyperstream::encoding::RandomProjectionEncoder;

// Simple 64-bit FNV-1a over words for debug/traceability.
static inline std::uint64_t HashWords(const std::uint64_t* words, std::size_t n) {
  const std::uint64_t kOffset = 1469598103934665603ull;
  const std::uint64_t kPrime = 1099511628211ull;
  std::uint64_t h = kOffset;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t w = words[i];
    for (int b = 0; b < 8; ++b) {
      std::uint8_t byte = static_cast<std::uint8_t>(w & 0xFF);
      h ^= byte;
      h *= kPrime;
      w >>= 8;
    }
  }
  return h;
}

TEST(EncoderDeterminism, ItemMemory_SameSeedSameOutput) {
  static constexpr std::size_t D = 256;
  const std::uint64_t seed = 0x123456789abcdef0ull;
  ItemMemory<D> im_a(seed);
  ItemMemory<D> im_b(seed);

  HyperVector<D, bool> hv_a, hv_b;
  im_a.EncodeId(42, &hv_a);
  im_b.EncodeId(42, &hv_b);

  ASSERT_EQ(hv_a.Words().size(), hv_b.Words().size());
  for (std::size_t i = 0; i < hv_a.Words().size(); ++i) {
    EXPECT_EQ(hv_a.Words()[i], hv_b.Words()[i]) << "word " << i;
  }

  const auto h = HashWords(hv_a.Words().data(), hv_a.Words().size());
  ::testing::Test::RecordProperty("ItemMemory/D256/seed", static_cast<long long>(seed));
  ::testing::Test::RecordProperty("ItemMemory/D256/hash", static_cast<long long>(h));

  // Assert canonical hash matches committed reference for this platform
  const std::string json = ReadTextFile(TestsDir() + "/golden/encoder_hashes.json");
  const std::string expect_hex = FindHashJson(json, "ItemMemory", static_cast<int>(D), PlatformId());
  ASSERT_FALSE(expect_hex.empty()) << "missing expected hash for platform: " << PlatformId();
  const std::string got_hex = Hex64(h);
  EXPECT_EQ(got_hex, expect_hex) << "ItemMemory/D" << D << " seed=" << std::hex << seed;
}

TEST(EncoderDeterminism, SymbolEncoder_RoleRotationDeterministic) {
  static constexpr std::size_t D = 256;
  SymbolEncoder<D> sym(0x9e3779b97f4a7c15ull);

  HyperVector<D, bool> hv_role0, hv_role7;
  sym.EncodeToken("alpha", &hv_role0);
  sym.EncodeTokenRole("alpha", 7, &hv_role7);

  // Rotating role0 by 7 should equal role7.
  HyperVector<D, bool> rotated;
  hyperstream::core::PermuteRotate(hv_role0, 7, &rotated);
  ASSERT_EQ(rotated.Words().size(), hv_role7.Words().size());
  for (std::size_t i = 0; i < rotated.Words().size(); ++i) {
    EXPECT_EQ(rotated.Words()[i], hv_role7.Words()[i]) << "word " << i;
  }

  const auto h = HashWords(hv_role7.Words().data(), hv_role7.Words().size());
  ::testing::Test::RecordProperty("SymbolEncoder/D256/hash", static_cast<long long>(h));

  const std::string json = ReadTextFile(TestsDir() + "/golden/encoder_hashes.json");
  const std::string expect_hex = FindHashJson(json, "SymbolEncoder", static_cast<int>(D), PlatformId());
  ASSERT_FALSE(expect_hex.empty());
  EXPECT_EQ(Hex64(h), expect_hex) << "SymbolEncoder/D" << D << " token=alpha role=7";
}

TEST(EncoderDeterminism, NumericEncoders_FixedSeedStable) {
  static constexpr std::size_t D = 256;
  ThermometerEncoder<D> therm(0.0, 1.0);
  RandomProjectionEncoder<D> proj(0x51ed2701f3a5c7b9ull /*seed*/);

  HyperVector<D, bool> hv_t0, hv_t1, hv_p;
  therm.Encode(0.5, &hv_t0);
  therm.Encode(0.5, &hv_t1);
  const float vec[4] = {1.0f, -2.0f, 0.5f, 7.0f};
  proj.Encode(vec, 4, &hv_p);

  // Same input to thermometer yields identical HV.
  ASSERT_EQ(hv_t0.Words().size(), hv_t1.Words().size());
  for (std::size_t i = 0; i < hv_t0.Words().size(); ++i) {
    EXPECT_EQ(hv_t0.Words()[i], hv_t1.Words()[i]);
  }

  const auto ht = HashWords(hv_t0.Words().data(), hv_t0.Words().size());
  const auto hp = HashWords(hv_p.Words().data(), hv_p.Words().size());
  ::testing::Test::RecordProperty("Thermometer/D256/hash", static_cast<long long>(ht));
  ::testing::Test::RecordProperty("RandomProjection/D256/hash", static_cast<long long>(hp));

  const std::string json = ReadTextFile(TestsDir() + "/golden/encoder_hashes.json");
  const std::string plat = PlatformId();
  ASSERT_EQ(Hex64(ht), FindHashJson(json, "Thermometer", static_cast<int>(D), plat));
  ASSERT_EQ(Hex64(hp), FindHashJson(json, "RandomProjection", static_cast<int>(D), plat));
}


// Disabled generator to dump canonical encoder hashes as JSON lines
TEST(EncoderDeterminism, DISABLED_DumpEncoderHashes) {
  static constexpr std::size_t D = 256;
  const std::string plat = PlatformId();

  // ItemMemory
  {
    const std::uint64_t seed = 0x123456789abcdef0ull;
    ItemMemory<D> im(seed);
    HyperVector<D, bool> hv; im.EncodeId(42, &hv);
    const auto h = HashWords(hv.Words().data(), hv.Words().size());
    std::printf("{\"encoder\":\"ItemMemory\",\"dim\":%d,\"seed\":\"0x%llx\",\"platform\":\"%s\",\"hash\":\"%s\"}\n",
               (int)D, (unsigned long long)seed, plat.c_str(), Hex64(h).c_str());
  }
  // SymbolEncoder
  {
    SymbolEncoder<D> sym(0x9e3779b97f4a7c15ull);
    HyperVector<D, bool> hv; sym.EncodeTokenRole("alpha", 7, &hv);
    const auto h = HashWords(hv.Words().data(), hv.Words().size());
    std::printf("{\"encoder\":\"SymbolEncoder\",\"dim\":%d,\"platform\":\"%s\",\"hash\":\"%s\"}\n",
               (int)D, plat.c_str(), Hex64(h).c_str());
  }
  // Thermometer
  {
    ThermometerEncoder<D> therm(0.0, 1.0);
    HyperVector<D, bool> hv; therm.Encode(0.5, &hv);
    const auto h = HashWords(hv.Words().data(), hv.Words().size());
    std::printf("{\"encoder\":\"Thermometer\",\"dim\":%d,\"platform\":\"%s\",\"hash\":\"%s\"}\n",
               (int)D, plat.c_str(), Hex64(h).c_str());
  }
  // RandomProjection
  {
    RandomProjectionEncoder<D> proj(0x51ed2701f3a5c7b9ull);
    const float vec[4] = {1.0f, -2.0f, 0.5f, 7.0f};
    HyperVector<D, bool> hv; proj.Encode(vec, 4, &hv);
    const auto h = HashWords(hv.Words().data(), hv.Words().size());
    std::printf("{\"encoder\":\"RandomProjection\",\"dim\":%d,\"seed\":\"0x%llx\",\"platform\":\"%s\",\"hash\":\"%s\"}\n",
               (int)D, (unsigned long long)0x51ed2701f3a5c7b9ull, plat.c_str(), Hex64(h).c_str());
  }
}

} // namespace

