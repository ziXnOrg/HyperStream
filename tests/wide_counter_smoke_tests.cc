#include "hyperstream/core/ops.hpp"
#include "gtest/gtest.h"

using hyperstream::core::BinaryBundler;

TEST(WideBundler, CounterTypeIs32Bits) {
  constexpr std::size_t D = 64;
  using CT = typename BinaryBundler<D>::counter_t;
  static_assert(sizeof(CT) == 4, "counter_t expected 32-bit in wide mode");
  EXPECT_EQ(sizeof(CT), 4u);
}

