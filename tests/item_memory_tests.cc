#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/item_memory.hpp"

namespace {

using hyperstream::core::HammingDistance;
using hyperstream::core::HyperVector;
using hyperstream::encoding::ItemMemory;

TEST(ItemMemory, DeterministicAcrossInstancesAndCalls) {
  static constexpr std::size_t D = 256;
  const std::uint64_t seed = 0x1234abcd9876fedcULL;
  ItemMemory<D> a(seed);
  ItemMemory<D> b(seed);

  const std::array<std::uint64_t, 5> ids = {1, 42, 1024, 65535, 0xabcdef01ULL};
  for (auto id : ids) {
    HyperVector<D, bool> hv1, hv2;
    a.EncodeId(id, &hv1);
    b.EncodeId(id, &hv2);
    ASSERT_EQ(hv1.Words().size(), hv2.Words().size());
    for (std::size_t w = 0; w < hv1.Words().size(); ++w) {
      EXPECT_EQ(hv1.Words()[w], hv2.Words()[w]) << "word index " << w;
    }
  }

  const std::array<std::string_view, 3> toks = {"alpha", "sensor-42", "β"};
  for (auto t : toks) {
    HyperVector<D, bool> hv1, hv2;
    a.EncodeToken(t, &hv1);
    a.EncodeToken(t, &hv2);
    for (std::size_t w = 0; w < hv1.Words().size(); ++w) {
      EXPECT_EQ(hv1.Words()[w], hv2.Words()[w]) << "token repeat; word index " << w;
    }
  }
}

TEST(ItemMemory, SeedIndependenceApproxHalfHamming) {
  static constexpr std::size_t D = 512;
  ItemMemory<D> a(0x1111111111111111ULL);
  ItemMemory<D> b(0x2222222222222222ULL);

  HyperVector<D, bool> h1, h2;
  a.EncodeToken("independent", &h1);
  b.EncodeToken("independent", &h2);

  const std::size_t dist = HammingDistance(h1, h2);
  const double frac = static_cast<double>(dist) / static_cast<double>(D);
  EXPECT_GT(frac, 0.40) << frac;
  EXPECT_LT(frac, 0.60) << frac;
}

TEST(ItemMemory, BitDensityReasonable) {
  static constexpr std::size_t D = 256;
  ItemMemory<D> im(0x9bdcafe123456789ULL);

  const int N = 200;
  double avg_ones = 0.0;
  for (int i = 0; i < N; ++i) {
    HyperVector<D, bool> hv;
    im.EncodeId(static_cast<std::uint64_t>(i * 1315423911u), &hv);
    int ones = 0;
    for (std::size_t bit = 0; bit < D; ++bit) {
      if (hv.GetBit(bit)) ++ones;
    }
    avg_ones += static_cast<double>(ones);
  }
  avg_ones /= static_cast<double>(N);
  const double frac = avg_ones / static_cast<double>(D);
  EXPECT_GT(frac, 0.40) << frac;
  EXPECT_LT(frac, 0.60) << frac;
}

TEST(ItemMemory, TrailingBitsMasked) {
  static constexpr std::size_t D = 130;  // not multiple of 64
  ItemMemory<D> im(0xdeadbeefcafebabeULL);
  HyperVector<D, bool> hv;
  im.EncodeId(12345, &hv);

  const auto& words = hv.Words();
  ASSERT_EQ(words.size(), 3u);
  const std::size_t valid_in_last = D % 64;  // 2 bits valid
  switch (valid_in_last) {
    case 0:
      break;
    default: {
      const std::uint64_t valid_mask =
          (valid_in_last == 64) ? ~0ULL : ((1ULL << valid_in_last) - 1ULL);
      const std::uint64_t invalid_mask = ~valid_mask;
      EXPECT_EQ((words.back() & invalid_mask), 0ULL) << std::hex << words.back();
    } break;
  }
}

}  // namespace
