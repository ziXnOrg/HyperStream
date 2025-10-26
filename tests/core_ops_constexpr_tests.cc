#include <gtest/gtest.h>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

using hyperstream::core::Bind;
using hyperstream::core::BundleAdd;
using hyperstream::core::BundlePairMajority;
using hyperstream::core::HammingDistance;
using hyperstream::core::HyperVector;
using hyperstream::core::PermuteRotate;

namespace {

// Compile-time checks (C++17 constexpr)
namespace ct {
// Bind on small binary HV
constexpr bool TestBind() {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> a{};
  HyperVector<D, bool> b{};
  HyperVector<D, bool> out{};
  // set a: bits 0,2; b: bits 2,3
  auto aa = a;
  aa.Clear();
  aa.SetBit(0, true);
  aa.SetBit(2, true);
  auto bb = b;
  bb.Clear();
  bb.SetBit(2, true);
  bb.SetBit(3, true);
  Bind(aa, bb, &out);
  return out.GetBit(0) && !out.GetBit(2) && out.GetBit(3);
}

static_assert(TestBind(), "constexpr Bind failed");

// PermuteRotate on binary HV
constexpr bool TestPermute() {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> in{};
  HyperVector<D, bool> out{};
  auto x = in;
  x.Clear();
  x.SetBit(1, true);
  PermuteRotate(x, 2, &out);
  return out.GetBit(3) && !out.GetBit(1);
}

static_assert(TestPermute(), "constexpr PermuteRotate failed");

// HammingDistance on binary HV
constexpr bool TestHamming() {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> a{};
  auto aa = a;
  aa.Clear();
  aa.SetBit(0, true);
  HyperVector<D, bool> b{};
  auto bb = b;
  bb.Clear();
  bb.SetBit(0, true);
  bb.SetBit(1, true);
  return HammingDistance(aa, bb) == 1u;
}

static_assert(TestHamming(), "constexpr HammingDistance failed");

// BundleAdd for numeric HV
constexpr bool TestBundleAdd() {
  constexpr std::size_t D = 4;
  HyperVector<D, int> a{};
  HyperVector<D, int> b{};
  HyperVector<D, int> out{};
  for (std::size_t i = 0; i < D; ++i) {
    a[i] = static_cast<int>(i);
    b[i] = 1;
  }
  BundleAdd(a, b, &out);
  for (std::size_t i = 0; i < D; ++i) {
    if (out[i] != static_cast<int>(i + 1)) return false;
  }
  return true;
}

static_assert(TestBundleAdd(), "constexpr BundleAdd failed");

// BundlePairMajority for binary two-vector majority (OR)
constexpr bool TestBundlePairMajority() {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> a{};
  auto aa = a;
  aa.Clear();
  aa.SetBit(0, true);
  HyperVector<D, bool> b{};
  auto bb = b;
  bb.Clear();
  bb.SetBit(1, true);
  HyperVector<D, bool> out{};
  BundlePairMajority(aa, bb, &out);
  return out.GetBit(0) && out.GetBit(1);
}

static_assert(TestBundlePairMajority(), "constexpr BundlePairMajority failed");
}  // namespace ct

// Runtime mirrors
TEST(ConstexprOpsRuntime, Bind) {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> a, b, out;
  a.Clear();
  b.Clear();
  out.Clear();
  a.SetBit(0, true);
  a.SetBit(2, true);
  b.SetBit(2, true);
  b.SetBit(3, true);
  Bind(a, b, &out);
  EXPECT_TRUE(out.GetBit(0));
  EXPECT_FALSE(out.GetBit(2));
  EXPECT_TRUE(out.GetBit(3));
}

TEST(ConstexprOpsRuntime, PermuteRotate) {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> in, out;
  in.Clear();
  out.Clear();
  in.SetBit(1, true);
  PermuteRotate(in, 2, &out);
  EXPECT_TRUE(out.GetBit(3));
  EXPECT_FALSE(out.GetBit(1));
}

TEST(ConstexprOpsRuntime, HammingDistance) {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> a, b;
  a.Clear();
  b.Clear();
  a.SetBit(0, true);
  b.SetBit(0, true);
  b.SetBit(1, true);
  EXPECT_EQ(HammingDistance(a, b), 1u);
}

TEST(ConstexprOpsRuntime, BundleAddNumeric) {
  constexpr std::size_t D = 4;
  HyperVector<D, int> a, b, out;
  for (std::size_t i = 0; i < D; ++i) {
    a[i] = static_cast<int>(i);
    b[i] = 1;
  }
  BundleAdd(a, b, &out);
  for (std::size_t i = 0; i < D; ++i) {
    EXPECT_EQ(out[i], static_cast<int>(i + 1));
  }
}

TEST(ConstexprOpsRuntime, BundlePairMajority) {
  constexpr std::size_t D = 8;
  HyperVector<D, bool> a, b, out;
  a.Clear();
  b.Clear();
  out.Clear();
  a.SetBit(0, true);
  b.SetBit(1, true);
  BundlePairMajority(a, b, &out);
  EXPECT_TRUE(out.GetBit(0));
  EXPECT_TRUE(out.GetBit(1));
}

}  // namespace
