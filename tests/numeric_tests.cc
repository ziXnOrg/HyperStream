#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/numeric.hpp"

namespace {

using hyperstream::core::HyperVector;
using hyperstream::encoding::RandomProjectionEncoder;
using hyperstream::encoding::ThermometerEncoder;

static int CountOnes128(const HyperVector<128, bool>& hv) {
  int c = 0;
  for (std::size_t i = 0; i < 128; ++i)
    if (hv.GetBit(i)) ++c;
  return c;
}

TEST(ThermometerEncoder, MonotonicAndBounds) {
  static constexpr std::size_t D = 128;
  ThermometerEncoder<D> enc(0.0, 10.0);

  HyperVector<D, bool> lo, mid, hi, below, above;
  enc.Encode(0.0, &lo);
  enc.Encode(5.0, &mid);
  enc.Encode(10.0, &hi);
  enc.Encode(-5.0, &below);  // clamps to 0
  enc.Encode(15.0, &above);  // clamps to Dim

  const int c_lo = CountOnes128(lo);
  const int c_mid = CountOnes128(mid);
  const int c_hi = CountOnes128(hi);
  const int c_below = CountOnes128(below);
  const int c_above = CountOnes128(above);

  EXPECT_EQ(c_below, 0);
  EXPECT_EQ(c_lo, 0);
  EXPECT_GT(c_mid, 0);
  EXPECT_EQ(c_hi, static_cast<int>(D));
  EXPECT_EQ(c_above, static_cast<int>(D));
  EXPECT_LE(c_lo, c_mid);
  EXPECT_LE(c_mid, c_hi);
}

TEST(RandomProjectionEncoder, DeterministicAndEmptyYieldsZeros) {
  static constexpr std::size_t D = 128;
  RandomProjectionEncoder<D> enc(0x9e3779b97f4a7c15ULL);

  std::vector<float> x = {1.0f, -2.0f, 0.5f, 0.0f, 3.0f};
  HyperVector<D, bool> a, b, empty;
  enc.Encode(x.data(), x.size(), &a);
  enc.Encode(x.data(), x.size(), &b);

  for (std::size_t w = 0; w < a.Words().size(); ++w) {
    EXPECT_EQ(a.Words()[w], b.Words()[w]) << w;
  }

  enc.Encode(nullptr, 0, &empty);
  int ones = 0;
  for (std::size_t i = 0; i < D; ++i)
    if (empty.GetBit(i)) ++ones;
  EXPECT_EQ(ones, 0);
}

TEST(RandomProjectionEncoder, SignInversionFlipsManyBits) {
  static constexpr std::size_t D = 256;
  RandomProjectionEncoder<D> enc(0x123456789abcdef0ULL);

  std::vector<float> x(32);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>(i + 1);
  std::vector<float> y(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) y[i] = -x[i];

  HyperVector<D, bool> hx, hy;
  enc.Encode(x.data(), x.size(), &hx);
  enc.Encode(y.data(), y.size(), &hy);

  // Many bits should flip; expect > 60% Hamming distance.
  std::size_t dist = hyperstream::core::HammingDistance(hx, hy);
  double frac = static_cast<double>(dist) / static_cast<double>(D);
  EXPECT_GT(frac, 0.60) << frac;
}

}  // namespace
