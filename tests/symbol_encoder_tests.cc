#include <cstdint>
#include <gtest/gtest.h>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/symbol.hpp"

namespace {

using hyperstream::core::HyperVector;
using hyperstream::core::PermuteRotate;
using hyperstream::encoding::SymbolEncoder;

TEST(SymbolEncoder, DeterministicAndRoleRotationEquivalence) {
  static constexpr std::size_t D = 256;
  const std::uint64_t seed = 0x51ed2701f3a5c7b9ULL;
  SymbolEncoder<D> enc(seed);

  HyperVector<D, bool> base, role5, rotated;
  enc.EncodeToken("sensor-42", &base);
  enc.EncodeTokenRole("sensor-42", 5, &role5);

  PermuteRotate(base, 5, &rotated);
  ASSERT_EQ(role5.Words().size(), rotated.Words().size());
  for (std::size_t w = 0; w < rotated.Words().size(); ++w) {
    EXPECT_EQ(role5.Words()[w], rotated.Words()[w]) << "word index " << w;
  }
}

TEST(SymbolEncoder, EncodeIdMatchesRepeatedCalls) {
  static constexpr std::size_t D = 128;
  SymbolEncoder<D> enc(0x9e3779b97f4a7c15ULL);
  HyperVector<D, bool> a, b;
  enc.EncodeId(1337, &a);
  enc.EncodeId(1337, &b);
  for (std::size_t w = 0; w < a.Words().size(); ++w) {
    EXPECT_EQ(a.Words()[w], b.Words()[w]) << w;
  }
}

}  // namespace
