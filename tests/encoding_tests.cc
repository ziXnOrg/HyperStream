#include <array>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/encoders.hpp"

namespace {

using hyperstream::core::BinaryBundler;
using hyperstream::core::Bind;
using hyperstream::core::HyperVector;
using hyperstream::core::PermuteRotate;
using hyperstream::encoding::HashEncoder;
using hyperstream::encoding::RandomBasisEncoder;
using hyperstream::encoding::SequentialNGramEncoder;
using hyperstream::encoding::UnaryIntensityEncoder;
using hyperstream::encoding::detail::BuildVanDerCorputOrder;
using hyperstream::encoding::detail::GenerateRandomHypervector;

template <std::size_t Dim>
int CountOnes(const HyperVector<Dim, bool>& hv) {
  int count = 0;
  for (std::size_t i = 0; i < Dim; ++i) {
    if (hv.GetBit(i)) {
      ++count;
    }
  }
  return count;
}

TEST(RandomBasisEncoder, DeterministicAcrossInstances) {
  static constexpr std::size_t kDim = 256;
  RandomBasisEncoder<kDim> encoder_a(0x1234abcd9876fedcULL);
  RandomBasisEncoder<kDim> encoder_b(0x1234abcd9876fedcULL);

  const std::array<std::uint64_t, 5> symbols = {3, 77, 1024, 4096, 65535};
  for (auto symbol : symbols) {
    encoder_a.Update(symbol);
    encoder_b.Update(symbol);
  }

  HyperVector<kDim, bool> hv_a;
  HyperVector<kDim, bool> hv_b;
  encoder_a.Finalize(&hv_a);
  encoder_b.Finalize(&hv_b);

  ASSERT_EQ(hv_a.Words().size(), hv_b.Words().size());
  for (std::size_t w = 0; w < hv_a.Words().size(); ++w) {
    EXPECT_EQ(hv_a.Words()[w], hv_b.Words()[w]) << "word index " << w;
  }
}

TEST(HashEncoder, SetsKBitsAndRespectsRolePermutation) {
  static constexpr std::size_t kDim = 128;
  static constexpr int kHashes = 6;
  HashEncoder<kDim> encoder({kHashes, 0xfeedface12345678ULL});

  HyperVector<kDim, bool> hv_base;
  HyperVector<kDim, bool> hv_role;
  encoder.EncodeToken({"sensor-42", 0}, &hv_base);
  encoder.EncodeToken({"sensor-42", 5}, &hv_role);

  EXPECT_EQ(CountOnes(hv_base), kHashes);
  EXPECT_EQ(CountOnes(hv_role), kHashes);

  HyperVector<kDim, bool> rotated;
  PermuteRotate(hv_base, 5, &rotated);
  for (std::size_t w = 0; w < hv_base.Words().size(); ++w) {
    EXPECT_EQ(hv_role.Words()[w], rotated.Words()[w]) << "word index " << w;
  }
}

TEST(UnaryIntensityEncoder, MajorityMatchesManualCounters) {
  static constexpr std::size_t kDim = 32;
  static constexpr std::size_t kMaxIntensity = 8;
  UnaryIntensityEncoder<kDim> encoder(kMaxIntensity);

  // Manual counter simulation.
  const auto order = BuildVanDerCorputOrder<kDim>();
  std::array<int, kDim> counters{};
  std::size_t phase = 0;

  auto apply_manual = [&](std::size_t intensity) {
    const std::size_t clamped = (intensity > kMaxIntensity) ? kMaxIntensity : intensity;
    std::array<bool, kDim> bits{};
    for (std::size_t i = 0; i < clamped && i < kDim; ++i) {
      const std::size_t idx = order[(phase + i) % kDim];
      bits[idx] = true;
    }
    for (std::size_t i = 0; i < kDim; ++i) {
      counters[i] += bits[i] ? 1 : -1;
    }
    phase = (phase + clamped) % kDim;
  };

  const std::array<std::size_t, 3> intensities = {3, 9, 2};
  for (auto value : intensities) {
    apply_manual(value);
    encoder.Update(value);
  }

  HyperVector<kDim, bool> hv;
  encoder.Finalize(&hv);
  for (std::size_t i = 0; i < kDim; ++i) {
    const bool expected = counters[i] >= 0;
    EXPECT_EQ(hv.GetBit(i), expected) << "bit index " << i;
  }
}

TEST(SequentialNGramEncoder, AggregatesMatchesManualComputation) {
  static constexpr std::size_t kDim = 128;
  static constexpr std::size_t kWindow = 3;
  const std::uint64_t seed = 0x9bdcafe123456789ULL;

  SequentialNGramEncoder<kDim, kWindow> encoder(seed);
  BinaryBundler<kDim> bundler_manual;
  bundler_manual.Reset();

  std::array<std::uint64_t, kWindow> history{};
  std::size_t head = 0;
  std::size_t count = 0;

  const std::array<std::uint64_t, 6> stream = {42, 99, 1234, 2048, 4096, 8192};
  for (auto symbol : stream) {
    history[head] = symbol;
    head = (head + 1) % kWindow;

    if (count < kWindow) {
      ++count;
      encoder.Update(symbol);
      continue;
    }

    // Encoder will process an n-gram on this update.
    encoder.Update(symbol);

    HyperVector<kDim, bool> aggregate;
    aggregate.Clear();
    bool first = true;
    for (std::size_t i = 0; i < kWindow; ++i) {
      const std::size_t idx = (head + kWindow - 1 - i) % kWindow;
      HyperVector<kDim, bool> hv;
      GenerateRandomHypervector(seed, history[idx], &hv);

      if (i != 0) {
        HyperVector<kDim, bool> rotated;
        PermuteRotate(hv, i, &rotated);
        hv = rotated;
      }

      if (first) {
        aggregate = hv;
        first = false;
      } else {
        HyperVector<kDim, bool> bound;
        Bind(aggregate, hv, &bound);
        aggregate = bound;
      }
    }
    bundler_manual.Accumulate(aggregate);
  }

  HyperVector<kDim, bool> expected;
  bundler_manual.Finalize(&expected);

  HyperVector<kDim, bool> actual;
  encoder.Finalize(&actual);

  ASSERT_EQ(expected.Words().size(), actual.Words().size());
  for (std::size_t w = 0; w < expected.Words().size(); ++w) {
    EXPECT_EQ(expected.Words()[w], actual.Words()[w]) << "word index " << w;
  }
}

}  // namespace
