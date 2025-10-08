#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/encoding/item_memory.hpp"
#include "hyperstream/encoding/symbol.hpp"
#include "hyperstream/encoding/numeric.hpp"

namespace {

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
}

} // namespace

