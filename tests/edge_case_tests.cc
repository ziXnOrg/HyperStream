// Edge case tests for HyperStream core operations.
// - Covers min dimension, non-power-of-two behavior, out-of-range exceptions,
//   and numeric/complex correctness with cosine similarity.
// - Uses explicit using declarations (no using namespace) for consistency with
//   tests/core_ops_tests.cc and common C++ style.
#include <gtest/gtest.h>
#include <complex>
#include <stdexcept>
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

using hyperstream::core::HyperVector;
using hyperstream::core::Bind;
using hyperstream::core::HammingDistance;
using hyperstream::core::CosineSimilarity;

namespace {

// Test with minimum dimension (8 bits)
TEST(EdgeCaseTests, MinDimension) {
    constexpr size_t kDim = 8;
    HyperVector<kDim, bool> a, b, result;
    
    // Test with all zeros
    for (size_t i = 0; i < kDim; ++i) {
        a.SetBit(i, false);
        b.SetBit(i, false);
    }
    
    // Test binding zeros
    Bind(a, b, &result);
    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_FALSE(result.GetBit(i)) << "Binding two zero vectors should result in zero vector";
    }
    
    // Test with all ones
    for (size_t i = 0; i < kDim; ++i) {
        a.SetBit(i, true);
        b.SetBit(i, true);
    }
    
    // Test binding ones
    Bind(a, b, &result);
    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_FALSE(result.GetBit(i)) << "Binding two one vectors should result in zero vector";
    }
}

// Test with non-power-of-two dimensions
TEST(EdgeCaseTests, NonPowerOfTwoDimensions) {
    constexpr size_t kDim = 100;  // Not a power of two
    HyperVector<kDim, bool> a, b, result;
    
    // Test with edge bits set
    a.SetBit(0, true);
    b.SetBit(kDim - 1, true);
    
    // Test binding
    Bind(a, b, &result);
    EXPECT_TRUE(result.GetBit(0));
    EXPECT_TRUE(result.GetBit(kDim - 1));
    
    // Test Hamming distance
    auto dist = HammingDistance(a, b);
    EXPECT_EQ(dist, 2u) << "Expected Hamming distance of 2 for vectors with first and last bits set";
}

// Test with different numeric types
TEST(EdgeCaseTests, NumericTypes) {
    constexpr size_t kDim = 32;
    
    // Test with different numeric types
    HyperVector<kDim, int8_t> int8_vec;
    HyperVector<kDim, uint8_t> uint8_vec;
    
    // Set and verify values
    for (size_t i = 0; i < kDim; ++i) {
        int8_vec[i] = static_cast<int8_t>(i);
        uint8_vec[i] = static_cast<uint8_t>(i);
    }
    
    // Verify values
    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_EQ(int8_vec[i], static_cast<int8_t>(i));
        EXPECT_EQ(uint8_vec[i], static_cast<uint8_t>(i));
    }
}

// Out-of-range access should throw std::out_of_range for bool specialization
TEST(EdgeCaseTests, OutOfRangeThrows) {
    constexpr size_t D = 8;
    HyperVector<D, bool> hv;
    EXPECT_THROW({ volatile bool sink = hv.GetBit(D); (void)sink; }, std::out_of_range);
    EXPECT_THROW(hv.SetBit(D, true), std::out_of_range);
}

// Float and complex types: value set/get and cosine similarity
TEST(EdgeCaseTests, FloatAndComplexTypes) {
    constexpr size_t D = 16;
    HyperVector<D, float> af, bf;
    HyperVector<D, std::complex<float>> ac, bc;
    for (size_t i = 0; i < D; ++i) {
        af[i] = static_cast<float>(i) * 0.5f;
        bf[i] = af[i];
        ac[i] = {1.0f, static_cast<float>(i) * 0.25f};
        bc[i] = ac[i];
    }
    // Exact value checks
    for (size_t i = 0; i < D; ++i) {
        EXPECT_FLOAT_EQ(af[i], static_cast<float>(i) * 0.5f);
        EXPECT_FLOAT_EQ(ac[i].real(), 1.0f);
        EXPECT_FLOAT_EQ(ac[i].imag(), static_cast<float>(i) * 0.25f);
    }
    // Cosine similarity ~ 1.0 for identical vectors
    EXPECT_NEAR(CosineSimilarity(af, bf), 1.0f, 1e-6f);
    EXPECT_NEAR(CosineSimilarity(ac, bc), 1.0f, 1e-6f);
}

// Large, but safe dimension to avoid excessive stack usage
TEST(EdgeCaseTests, MaxReasonableDimension_BasicOps) {
    constexpr size_t D = 1 << 16; // 65,536
    HyperVector<D, bool> a, b, out;
    a.Clear(); b.Clear(); out.Clear();
    a.SetBit(0, true);
    b.SetBit(D - 1, true);
    Bind(a, b, &out);
    EXPECT_TRUE(out.GetBit(0));
    EXPECT_TRUE(out.GetBit(D - 1));
    EXPECT_EQ(HammingDistance(a, b), 2u);
}

}  // namespace
