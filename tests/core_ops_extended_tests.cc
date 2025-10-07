#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

using hyperstream::core::HyperVector;
using hyperstream::core::BinaryBundler;
using hyperstream::core::Bind;
using hyperstream::core::PermuteRotate;
using hyperstream::core::HammingDistance;
using hyperstream::core::NormalizedHammingSimilarity;
using hyperstream::core::CosineSimilarity;

namespace {

// Test fixture for parameterized tests
class CoreOpsTest : public ::testing::Test {
protected:
    static constexpr size_t kTestDimension = 1024;
    
    void SetUp() override {
        // Initialize random number generation
        std::mt19937 gen(42);
        std::bernoulli_distribution dist(0.5);

        // Generate random hypervectors
        for (size_t i = 0; i < kTestDimension; ++i) {
            hv1_.SetBit(i, dist(gen));
            hv2_.SetBit(i, dist(gen));
            hv3_.SetBit(i, dist(gen));
        }
    }
    
    HyperVector<kTestDimension, bool> hv1_, hv2_, hv3_;
    HyperVector<kTestDimension, bool> result_;
};

// Test binding operation properties
TEST_F(CoreOpsTest, BindingProperties) {
    // Test identity: a ^ a = 0
    Bind(hv1_, hv1_, &result_);
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_FALSE(result_.GetBit(i)) << "XOR of a vector with itself should be zero";
    }
    
    // Test commutativity: a ^ b = b ^ a
    HyperVector<kTestDimension, bool> result2;
    Bind(hv1_, hv2_, &result_);
    Bind(hv2_, hv1_, &result2);
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(result_.GetBit(i), result2.GetBit(i)) 
            << "Binding operation should be commutative";
    }
    
    // Test associativity: (a ^ b) ^ c = a ^ (b ^ c)
    HyperVector<kTestDimension, bool> temp1, temp2;
    Bind(hv1_, hv2_, &temp1);
    Bind(temp1, hv3_, &result_);
    
    Bind(hv2_, hv3_, &temp2);
    Bind(hv1_, temp2, &result2);
    
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(result_.GetBit(i), result2.GetBit(i))
            << "Binding operation should be associative";
    }
}

// Test bundling operation properties
TEST_F(CoreOpsTest, BundlingProperties) {
    BinaryBundler<kTestDimension> bundler;
    
    // Test that bundling with a single vector returns the same vector
    bundler.Reset();
    bundler.Accumulate(hv1_);
    bundler.Finalize(&result_);
    
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(hv1_.GetBit(i), result_.GetBit(i))
            << "Bundling a single vector should return the same vector";
    }
    
    // Test that bundling is commutative
    bundler.Reset();
    bundler.Accumulate(hv1_);
    bundler.Accumulate(hv2_);
    bundler.Finalize(&result_);
    
    HyperVector<kTestDimension, bool> result2;
    bundler.Reset();
    bundler.Accumulate(hv2_);
    bundler.Accumulate(hv1_);
    bundler.Finalize(&result2);
    
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(result_.GetBit(i), result2.GetBit(i))
            << "Bundling operation should be commutative";
    }
}

// Test permutation operation properties
TEST_F(CoreOpsTest, PermutationProperties) {
    // Test that permutation by 0 is identity
    PermuteRotate(hv1_, 0, &result_);
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(hv1_.GetBit(i), result_.GetBit(i))
            << "Permutation by 0 should be identity";
    }
    
    // Test that permutation by dimension is identity (full rotation)
    PermuteRotate(hv1_, kTestDimension, &result_);
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(hv1_.GetBit(i), result_.GetBit(i))
            << "Permutation by dimension should be identity";
    }
    
    // Test composition of permutations
    const size_t shift1 = 5;
    const size_t shift2 = 7;
    const size_t total_shift = (shift1 + shift2) % kTestDimension;
    
    PermuteRotate(hv1_, shift1, &result_);
    PermuteRotate(result_, shift2, &result_);
    
    HyperVector<kTestDimension, bool> expected;
    PermuteRotate(hv1_, total_shift, &expected);
    
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(expected.GetBit(i), result_.GetBit(i))
            << "Permutations should compose correctly";
    }
}

// Test similarity measures
TEST_F(CoreOpsTest, SimilarityMeasures) {
    // Test Hamming distance properties
    const auto dist_ab = HammingDistance(hv1_, hv2_);
    const auto dist_ba = HammingDistance(hv2_, hv1_);
    EXPECT_EQ(dist_ab, dist_ba) << "Hamming distance should be symmetric";
    
    // Test that distance to self is zero
    const auto dist_aa = HammingDistance(hv1_, hv1_);
    EXPECT_EQ(dist_aa, 0u) << "Hamming distance from a vector to itself should be zero";
    
    // Test normalized similarity bounds
    const float sim_ab = NormalizedHammingSimilarity(hv1_, hv2_);
    EXPECT_GE(sim_ab, -1.0f) << "Similarity should be >= -1";
    EXPECT_LE(sim_ab, 1.0f) << "Similarity should be <= 1";
    
    // Test that similarity to self is 1.0
    const float sim_aa = NormalizedHammingSimilarity(hv1_, hv1_);
    EXPECT_NEAR(sim_aa, 1.0f, 1e-6f) << "Similarity to self should be 1.0";
}

// Test edge cases
TEST_F(CoreOpsTest, EdgeCases) {
    // Test with all zeros
    HyperVector<kTestDimension, bool> zero;
    zero.Clear();
    
    // Binding with zero vector should be identity
    Bind(hv1_, zero, &result_);
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_EQ(hv1_.GetBit(i), result_.GetBit(i))
            << "Binding with zero vector should be identity";
    }
    
    // Test with all ones
    HyperVector<kTestDimension, bool> ones;
    for (size_t i = 0; i < kTestDimension; ++i) {
        ones.SetBit(i, true);
    }
    
    // Binding with ones should be bitwise NOT
    Bind(hv1_, ones, &result_);
    for (size_t i = 0; i < kTestDimension; ++i) {
        EXPECT_NE(hv1_.GetBit(i), result_.GetBit(i))
            << "Binding with ones should flip all bits";
    }
    
    // Test with single-bit vectors
    for (size_t i = 0; i < 8; ++i) {  // Test first 8 bits
        HyperVector<kTestDimension, bool> single_bit;
        single_bit.Clear();
        single_bit.SetBit(i, true);
        
        // Binding two single-bit vectors should have exactly 2 bits set (unless i == j)
        for (size_t j = 0; j < 8; ++j) {
            HyperVector<kTestDimension, bool> other_bit;
            other_bit.Clear();
            other_bit.SetBit(j, true);
            
            Bind(single_bit, other_bit, &result_);
            
            size_t count = 0;
            for (size_t k = 0; k < kTestDimension; ++k) {
                if (result_.GetBit(k)) count++;
            }
            
            if (i == j) {
                EXPECT_EQ(count, 0u) << "Binding a bit with itself should result in zero";
            } else {
                EXPECT_EQ(count, 2u) << "Binding two different bits should result in 2 bits set";
            }
        }
    }
}

// Test with different hypervector dimensions
TEST(HyperVectorDimensionality, VariousDimensions) {
    // Test with small dimension
    {
        constexpr size_t small_dim = 8;
        HyperVector<small_dim, bool> a, b, result;
        a.Clear(); b.Clear();
        
        a.SetBit(0, true);
        b.SetBit(0, true);
        b.SetBit(1, true);
        
        Bind(a, b, &result);
        EXPECT_FALSE(result.GetBit(0));
        EXPECT_TRUE(result.GetBit(1));
        
        const auto dist = HammingDistance(a, b);
        EXPECT_EQ(dist, 1u);
    }
    
    // Test with dimension not a multiple of 64
    {
        constexpr size_t odd_dim = 100;
        HyperVector<odd_dim, bool> a, b, result;
        a.Clear(); b.Clear();
        
        a.SetBit(0, true);
        b.SetBit(odd_dim - 1, true);
        
        Bind(a, b, &result);
        EXPECT_TRUE(result.GetBit(0));
        EXPECT_TRUE(result.GetBit(odd_dim - 1));
        
        const auto dist = HammingDistance(a, b);
        EXPECT_EQ(dist, 2u);
    }
}

}  // namespace
