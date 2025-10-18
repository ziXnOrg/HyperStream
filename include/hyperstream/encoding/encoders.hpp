#pragma once

// =============================================================================
// File:        include/hyperstream/encoding/encoders.hpp
// Overview:    Binary encoders implementing HyperStream streaming APIs.
//              Provides random-basis, hash-based, unary intensity, and sequential
//              n-gram encoders. Encoders avoid dynamic allocation and expose
//              Reset/Update/Finalize for streaming.
// Mathematical Foundation: Low-discrepancy sequences (van der Corput),
//              64-bit SplitMix variants, FNV-1a hashing, double hashing for bit
//              selection; majority bundling for aggregation.
// Security Considerations: Deterministic hashing/PRNG; seeds are explicit and
//              must be treated as non-cryptographic. No dynamic allocations in
//              hot paths; all inputs validated by types/limits.
// Performance Considerations: O(Dim/64) per update; contiguous storage; no
//              exceptions; avoids heap; tail-bit masking for non-multiple-of-64.
// Examples:    RandomBasisEncoder, HashEncoder, UnaryIntensityEncoder,
//              SequentialNGramEncoder.
// =============================================================================

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

namespace hyperstream::encoding {

namespace detail {

constexpr std::uint64_t kGoldenGamma = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kSplitMixMul1 = 0xbf58476d1ce4e5b9ULL;
constexpr std::uint64_t kSplitMixMul2 = 0x94d049bb133111ebULL;
constexpr std::uint32_t kSplitMixShift1 = 30U;
constexpr std::uint32_t kSplitMixShift2 = 27U;
constexpr std::uint32_t kSplitMixShift3 = 31U;
constexpr std::uint32_t kHalfWordBits = 32U;
constexpr std::uint64_t kFnvOffsetBasis64 = 1469598103934665603ULL;
constexpr std::uint64_t kFnvPrime64 = 1099511628211ULL;
constexpr std::uint64_t kTokenSalt = 0x5bf03635f0b7a54dULL;
constexpr std::uint64_t kRandomBasisDefaultSeed = 0x9e3779b97f4a7c15ULL;
constexpr std::uint64_t kSequentialNGramDefaultSeed = 0x27d4eb2f165667c5ULL;
constexpr std::uint64_t kHashEncoderDefaultSeed = 0x51ed2701f3a5c7b9ULL;

[[nodiscard]] inline std::uint64_t SplitMix64Step(std::uint64_t& state) noexcept {
  state += kGoldenGamma;
  std::uint64_t mixed_value = state;
  mixed_value = (mixed_value ^ (mixed_value >> kSplitMixShift1)) * kSplitMixMul1;
  mixed_value = (mixed_value ^ (mixed_value >> kSplitMixShift2)) * kSplitMixMul2;
  return mixed_value ^ (mixed_value >> kSplitMixShift3);
}

[[nodiscard]] inline std::uint64_t MixSymbol(std::uint64_t seed_value, std::uint64_t symbol_value) noexcept {
  std::uint64_t state = seed_value + (symbol_value * kSplitMixMul2);
  state ^= (symbol_value << kHalfWordBits) | (symbol_value >> kHalfWordBits);
  state *= kSplitMixMul1;
  return state;
}

struct RandomHypervectorInputs {
  std::uint64_t seed_value;
  std::uint64_t symbol_value;
};

template <std::size_t Dim>
inline void GenerateRandomHypervector(const RandomHypervectorInputs& inputs,
                                      core::HyperVector<Dim, bool>* out) noexcept {
  auto& words = out->Words();
  std::uint64_t state = MixSymbol(inputs.seed_value, inputs.symbol_value);
  for (std::size_t word_index = 0; word_index < words.size(); ++word_index) {
    words[word_index] = SplitMix64Step(state);
  }
  // Mask trailing bits beyond Dim.
  constexpr std::size_t kWordBits = core::HyperVector<Dim, bool>::kWordBits;
  const std::size_t excess_bits = words.size() * kWordBits - Dim;
  if (excess_bits > 0) {
    const std::uint64_t tail_mask = (~0ULL) >> excess_bits;
    words.back() &= tail_mask;
  }
}

  // Overload for legacy tests: (seed, symbol, out)
  template <std::size_t Dim>
  inline void GenerateRandomHypervector(std::uint64_t seed_value,
                                        std::uint64_t symbol_value,
                                        core::HyperVector<Dim, bool>* out) noexcept {
    GenerateRandomHypervector<Dim>({seed_value, symbol_value}, out);
  }


struct Fnv1a64Args {
  std::string_view token;
  std::uint64_t seed_value;
};

[[nodiscard]] inline std::uint64_t Fnv1a64(const Fnv1a64Args& args) noexcept {
  std::uint64_t hash = kFnvOffsetBasis64 ^ args.seed_value;
  for (unsigned char char_value : args.token) {
    hash ^= static_cast<std::uint64_t>(char_value);
    hash *= kFnvPrime64;
  }
  return hash;
}

struct DoubleHashArgs {
  std::string_view token;
  std::uint64_t seed_value;
};

[[nodiscard]] inline std::pair<std::uint64_t, std::uint64_t> DoubleHash(const DoubleHashArgs& args) noexcept {
  const std::uint64_t hash1 = Fnv1a64({args.token, args.seed_value});
  std::uint64_t hash2 = Fnv1a64({args.token, args.seed_value ^ kTokenSalt});
  hash2 = (hash2 << 1U) | 1ULL;  // ensure odd step
  return {hash1, hash2};
}

template <std::size_t Dim>
[[nodiscard]] inline std::array<std::size_t, Dim> BuildVanDerCorputOrder() noexcept {
  std::array<std::size_t, Dim> order{};
  for (std::size_t index = 0; index < Dim; ++index) {
    std::size_t reversed_bits_value = 0;
    std::size_t remaining_bits = index;
    static constexpr std::uint64_t kGrowthFactor = 2ULL;
    for (std::size_t bit_index = 0; (1ULL << bit_index) <= Dim * kGrowthFactor; ++bit_index) {
      reversed_bits_value = (reversed_bits_value << 1U) | (remaining_bits & 1ULL);
      remaining_bits >>= 1U;
    }
    order[index] = reversed_bits_value % Dim;
  }
  // Ensure permutation: fallback to identity for duplicates.
  std::array<bool, Dim> used{};
  for (std::size_t index = 0; index < Dim; ++index) {
    std::size_t mapped_index = order[index];
    if (mapped_index >= Dim || used[mapped_index]) {
      mapped_index = 0;
      while (used[mapped_index]) {
        mapped_index++;
      }
      order[index] = mapped_index;
    }
    used[order[index]] = true;
  }
  return order;
}

}  // namespace detail

// Random basis encoder with permutation per timestep and majority bundling.
template <std::size_t Dim>
class RandomBasisEncoder {
 public:
  explicit RandomBasisEncoder(std::uint64_t seed = detail::kRandomBasisDefaultSeed)
      : seed_(seed), step_(0) {
    Reset();
  }

  void Reset() noexcept {
    bundler_.Reset();
    step_ = 0;
  }

  void Update(std::uint64_t symbol) noexcept {
    core::HyperVector<Dim, bool> hv;
    hv.Clear();
    detail::GenerateRandomHypervector<Dim>({seed_, symbol}, &hv);
    if (step_ != 0) {
      core::HyperVector<Dim, bool> rotated;
      rotated.Clear();
      core::PermuteRotate(hv, step_, &rotated);
      bundler_.Accumulate(rotated);
    } else {
      bundler_.Accumulate(hv);
    }
    step_ = (step_ + 1) % Dim;
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const noexcept {
    bundler_.Finalize(out);
  }

 private:
  std::uint64_t seed_;
  std::size_t step_;
  core::BinaryBundler<Dim> bundler_;
};

// Hash/streaming encoder using double hashing to set K positions.
struct HashEncoderConfig {
  int num_hashes;
  std::uint64_t seed;
};

static constexpr int kDefaultNumHashes = 4;
template <std::size_t Dim>
class HashEncoder {
 public:
  explicit HashEncoder(HashEncoderConfig config = {kDefaultNumHashes, detail::kHashEncoderDefaultSeed})
      : k_(config.num_hashes), seed_(config.seed) {
    Reset();
  }

  void Reset() noexcept {
    bundler_.Reset();
  }

  struct TokenRole {
    std::string_view token;
    std::size_t role_index{0};
  };

  void Update(TokenRole token_role) noexcept {
    core::HyperVector<Dim, bool> hv;
    hv.Clear();
    EncodeToken(token_role, &hv);
    bundler_.Accumulate(hv);
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const noexcept {
    bundler_.Finalize(out);
  }

  void EncodeToken(TokenRole token_role, core::HyperVector<Dim, bool>* out) const noexcept {
    out->Clear();
    const auto [hash1, hash2] = detail::DoubleHash({token_role.token, seed_});
    for (int hash_index = 0; hash_index < k_; ++hash_index) {
      const std::size_t bit_pos = static_cast<std::size_t>(
          (hash1 + static_cast<std::uint64_t>(hash_index) * hash2) % Dim);
      out->SetBit(bit_pos, true);
    }
    if (token_role.role_index != 0) {
      core::HyperVector<Dim, bool> rotated;
      rotated.Clear();
      core::PermuteRotate(*out, token_role.role_index, &rotated);
      *out = rotated;
    }
  }

 private:
  int k_;
  std::uint64_t seed_;
  core::BinaryBundler<Dim> bundler_;
};

// Unary intensity encoder with low-discrepancy bit assignment.
template <std::size_t Dim>
class UnaryIntensityEncoder {
 public:
  explicit UnaryIntensityEncoder(std::size_t max_intensity)
      : max_intensity_(max_intensity), order_(detail::BuildVanDerCorputOrder<Dim>()), phase_(0) {
    Reset();
  }

  void Reset() noexcept {
    bundler_.Reset();
    phase_ = 0;
  }

  void Update(std::size_t intensity) noexcept {
    const std::size_t clamped = (intensity > max_intensity_) ? max_intensity_ : intensity;
    core::HyperVector<Dim, bool> hv;
    hv.Clear();
    for (std::size_t bit_index = 0; bit_index < clamped && bit_index < Dim; ++bit_index) {
      const std::size_t index = order_[(phase_ + bit_index) % Dim];
      hv.SetBit(index, true);
    }
    bundler_.Accumulate(hv);
    phase_ = (phase_ + clamped) % Dim;
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const noexcept {
    bundler_.Finalize(out);
  }

 private:
  std::size_t max_intensity_;
  std::array<std::size_t, Dim> order_;
  std::size_t phase_;
  core::BinaryBundler<Dim> bundler_;
};

// Sequential n-gram encoder binding permuted symbol hypervectors over a sliding window.
template <std::size_t Dim, std::size_t Window>
class SequentialNGramEncoder {
 public:
  static_assert(Window > 0, "SequentialNGramEncoder requires Window > 0");

  explicit SequentialNGramEncoder(std::uint64_t seed = detail::kSequentialNGramDefaultSeed)
      : seed_(seed), head_(0), count_(0) {
    Reset();
  }

  void Reset() noexcept {
    bundler_.Reset();
    head_ = 0;
    count_ = 0;
  }

  void Update(std::uint64_t symbol) noexcept {
    history_[head_] = symbol;
    head_ = (head_ + 1) % Window;
    if (count_ < Window) {
      ++count_;
      return;  // wait until full window to form n-gram
    }

    core::HyperVector<Dim, bool> aggregate;
    aggregate.Clear();
    bool is_first = true;
    for (std::size_t offset = 0; offset < Window; ++offset) {
      const std::size_t idx = (head_ + Window - 1 - offset) % Window;
      core::HyperVector<Dim, bool> hv;
      hv.Clear();
      detail::GenerateRandomHypervector(seed_, history_[idx], &hv);
      if (offset != 0) {
        core::HyperVector<Dim, bool> rotated;
        rotated.Clear();
        core::PermuteRotate(hv, offset, &rotated);
        hv = rotated;
      }
      if (is_first) {
        aggregate = hv;
        is_first = false;
      } else {
        core::HyperVector<Dim, bool> bound;
        bound.Clear();
        core::Bind(aggregate, hv, &bound);
        aggregate = bound;
      }
    }
    bundler_.Accumulate(aggregate);
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const noexcept {
    bundler_.Finalize(out);
  }

 private:
  std::uint64_t seed_;
  std::array<std::uint64_t, Window> history_{};
  std::size_t head_;
  std::size_t count_;
  core::BinaryBundler<Dim> bundler_;
};

}  // namespace hyperstream::encoding
