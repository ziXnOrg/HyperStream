#include <gtest/gtest.h>
#include <random>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

using hyperstream::core::BinaryBundler;
using hyperstream::core::Bind;
using hyperstream::core::CosineSimilarity;
using hyperstream::core::HammingDistance;
using hyperstream::core::HyperVector;
using hyperstream::core::NormalizedHammingSimilarity;
using hyperstream::core::PermuteRotate;

namespace {

TEST(HyperVectorBinary, GetSetBits) {
  constexpr std::size_t D = 128;
  HyperVector<D, bool> hv;
  hv.Clear();
  hv.SetBit(3, true);
  hv.SetBit(64, true);
  EXPECT_TRUE(hv.GetBit(3));
  EXPECT_TRUE(hv.GetBit(64));
  EXPECT_FALSE(hv.GetBit(5));
}

TEST(BindingBinary, XorBinding) {
  constexpr std::size_t D = 64;
  HyperVector<D, bool> a, b, out;
  a.Clear();
  b.Clear();
  out.Clear();
  a.SetBit(1, true);
  b.SetBit(1, true);
  b.SetBit(2, true);
  Bind(a, b, &out);
  EXPECT_TRUE(out.GetBit(2));
  EXPECT_FALSE(out.GetBit(1));
}

TEST(BundlingBinary, MajorityCounters) {
  constexpr std::size_t D = 32;
  HyperVector<D, bool> x1, x2, out;
  x1.Clear();
  x2.Clear();
  // x1 has bits 0..15 set; x2 has bits 8..23 set
  for (std::size_t i = 0; i < 16; ++i) x1.SetBit(i, true);
  for (std::size_t i = 8; i < 24 && i < D; ++i) x2.SetBit(i, true);
  BinaryBundler<D> bundler;
  bundler.Reset();
  bundler.Accumulate(x1);
  bundler.Accumulate(x2);
  bundler.Finalize(&out);
  // Majority expects 0..7 set by x1-only, 8..15 set by both, 16..23 set by x2-only
  for (std::size_t i = 0; i < 8; ++i) EXPECT_TRUE(out.GetBit(i));
  for (std::size_t i = 8; i < 16; ++i) EXPECT_TRUE(out.GetBit(i));
  for (std::size_t i = 16; i < 24 && i < D; ++i) EXPECT_TRUE(out.GetBit(i));
}

TEST(PermutationBinary, RotateLeft) {
  constexpr std::size_t D = 32;
  HyperVector<D, bool> in, out;
  in.Clear();
  in.SetBit(0, true);
  PermuteRotate(in, 5, &out);
  EXPECT_TRUE(out.GetBit(5));
}

TEST(SimilarityBinary, HammingAndNormalized) {
  constexpr std::size_t D = 64;
  HyperVector<D, bool> a, b;
  a.Clear();
  b.Clear();
  a.SetBit(0, true);
  b.SetBit(0, true);
  b.SetBit(1, true);
  EXPECT_EQ(HammingDistance(a, b), 1u);
  const float sim = NormalizedHammingSimilarity<D>({&a, &b});
  // When D=64 and hamming=1, sim = 1 - 2/64 = 0.96875
  EXPECT_NEAR(sim, 0.96875f, 1e-6f);
}

TEST(SimilarityComplex, Cosine) {
  constexpr std::size_t D = 4;
  HyperVector<D, std::complex<float>> a, b;
  for (std::size_t i = 0; i < D; ++i) {
    a[i] = {1.0f, 0.0f};
    b[i] = {1.0f, 0.0f};
  }
  const float sim = CosineSimilarity<D, std::complex<float>>({&a, &b});
  EXPECT_NEAR(sim, 1.0f, 1e-6f);
}

}  // namespace

TEST(BundlingBinary, SaturatingCountersUp) {
  constexpr std::size_t D = 64;
  HyperVector<D, bool> ones, out;
  ones.Clear();
  for (std::size_t i = 0; i < D; ++i) ones.SetBit(i, true);
  BinaryBundler<D> bundler;
  bundler.Reset();
  // Exceed int16 max to ensure saturation does not overflow
  for (int it = 0; it < 40000; ++it) bundler.Accumulate(ones);
  bundler.Finalize(&out);
  for (std::size_t i = 0; i < D; ++i) EXPECT_TRUE(out.GetBit(i));
}

TEST(BundlingBinary, SaturatingCountersDown) {
  constexpr std::size_t D = 64;
  HyperVector<D, bool> zeros, out;
  zeros.Clear();  // all bits false -> contributes -1 per bit
  BinaryBundler<D> bundler;
  bundler.Reset();
  for (int it = 0; it < 40000; ++it) bundler.Accumulate(zeros);
  bundler.Finalize(&out);
  for (std::size_t i = 0; i < D; ++i) EXPECT_FALSE(out.GetBit(i));
}

TEST(PropertyCoreOps, BindInvertibility_FixedSeed) {
  constexpr std::size_t D = 256;
  HyperVector<D, bool> a, key, bound, recovered;
  a.Clear();
  key.Clear();
  std::mt19937 gen(42);
  std::bernoulli_distribution dist(0.5);
  for (std::size_t i = 0; i < D; ++i) {
    if (dist(gen)) a.SetBit(i, true);
    if (dist(gen)) key.SetBit(i, true);
  }
  Bind(a, key, &bound);
  Bind(bound, key, &recovered);
  for (std::size_t w = 0; w < recovered.Words().size(); ++w) {
    EXPECT_EQ(recovered.Words()[w], a.Words()[w]) << "word index " << w;
  }
}

TEST(PropertyCoreOps, HammingTriangleInequality_FixedSeed) {
  constexpr std::size_t D = 256;
  HyperVector<D, bool> a, b, c;
  a.Clear();
  b.Clear();
  c.Clear();
  std::mt19937 gen(42);
  std::bernoulli_distribution dist(0.5);
  for (std::size_t i = 0; i < D; ++i) {
    if (dist(gen)) a.SetBit(i, true);
    if (dist(gen)) b.SetBit(i, true);
    if (dist(gen)) c.SetBit(i, true);
  }
  const auto d_ab = HammingDistance(a, b);
  const auto d_bc = HammingDistance(b, c);
  const auto d_ac = HammingDistance(a, c);
  EXPECT_LE(d_ac, d_ab + d_bc);
}
