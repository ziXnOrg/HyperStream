#pragma once

// Binary encoders implementing HyperStream streaming APIs.
// Provides Random basis, hash-based, unary intensity, and sequential n-gram encoders.
// Encoders avoid dynamic allocation, expose Reset/Update/Finalize, and operate on
// `hyperstream::core::HyperVector<Dim, bool>`.

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

namespace hyperstream {
namespace encoding {

namespace detail {

constexpr std::uint64_t kGoldenGamma = 0x9e3779b97f4a7c15ULL;

inline std::uint64_t SplitMix64Step(std::uint64_t& state) {
  state += kGoldenGamma;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

inline std::uint64_t MixSymbol(std::uint64_t seed, std::uint64_t symbol) {
  std::uint64_t state = seed + symbol * 0x94d049bb133111ebULL;
  state ^= (symbol << 32) | (symbol >> 32);
  state *= 0xbf58476d1ce4e5b9ULL;
  return state;
}

template <std::size_t Dim>
inline void GenerateRandomHypervector(std::uint64_t seed, std::uint64_t symbol,
                                      core::HyperVector<Dim, bool>* out) {
  auto& words = out->Words();
  std::uint64_t state = MixSymbol(seed, symbol);
  for (std::size_t i = 0; i < words.size(); ++i) {
    words[i] = SplitMix64Step(state);
  }
  // Mask trailing bits beyond Dim.
  constexpr std::size_t kWordBits = core::HyperVector<Dim, bool>::kWordBits;
  const std::size_t excess = words.size() * kWordBits - Dim;
  if (excess > 0) {
    const std::uint64_t mask = (excess == kWordBits) ? 0ULL : (~0ULL >> excess);
    words.back() &= mask;
  }
}

inline std::uint64_t Fnv1a64(std::string_view token, std::uint64_t seed) {
  std::uint64_t hash = 1469598103934665603ULL ^ seed;
  for (unsigned char c : token) {
    hash ^= static_cast<std::uint64_t>(c);
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline std::pair<std::uint64_t, std::uint64_t> DoubleHash(std::string_view token,
                                                          std::uint64_t seed) {
  const std::uint64_t h1 = Fnv1a64(token, seed);
  std::uint64_t h2 = Fnv1a64(token, seed ^ 0x5bf03635f0b7a54dULL);
  h2 = (h2 << 1) | 1ULL;  // ensure odd step
  return {h1, h2};
}

template <std::size_t Dim>
inline std::array<std::size_t, Dim> BuildVanDerCorputOrder() {
  std::array<std::size_t, Dim> order{};
  for (std::size_t i = 0; i < Dim; ++i) {
    std::size_t value = 0;
    std::size_t bits = i;
    for (std::size_t b = 0; (1ULL << b) <= Dim * 2; ++b) {
      value = (value << 1) | (bits & 1ULL);
      bits >>= 1U;
    }
    order[i] = value % Dim;
  }
  // Ensure permutation: fallback to identity for duplicates.
  std::array<bool, Dim> used{};
  for (std::size_t i = 0; i < Dim; ++i) {
    std::size_t idx = order[i];
    if (idx >= Dim || used[idx]) {
      idx = 0;
      while (used[idx]) idx++;
      order[i] = idx;
    }
    used[order[i]] = true;
  }
  return order;
}

}  // namespace detail

// Random basis encoder with permutation per timestep and majority bundling.
template <std::size_t Dim>
class RandomBasisEncoder {
 public:
  explicit RandomBasisEncoder(std::uint64_t seed = 0x9e3779b97f4a7c15ULL) : seed_(seed), step_(0) {
    Reset();
  }

  void Reset() {
    bundler_.Reset();
    step_ = 0;
  }

  void Update(std::uint64_t symbol) {
    core::HyperVector<Dim, bool> hv;
    hv.Clear();
    detail::GenerateRandomHypervector(seed_, symbol, &hv);
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

  void Finalize(core::HyperVector<Dim, bool>* out) const {
    bundler_.Finalize(out);
  }

 private:
  std::uint64_t seed_;
  std::size_t step_;
  core::BinaryBundler<Dim> bundler_;
};

// Hash/streaming encoder using double hashing to set K positions.
template <std::size_t Dim>
class HashEncoder {
 public:
  explicit HashEncoder(int k = 4, std::uint64_t seed = 0x51ed2701f3a5c7b9ULL) : k_(k), seed_(seed) {
    Reset();
  }

  void Reset() {
    bundler_.Reset();
  }

  void Update(std::string_view token, std::size_t role = 0) {
    core::HyperVector<Dim, bool> hv;
    hv.Clear();
    EncodeToken(token, role, &hv);
    bundler_.Accumulate(hv);
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const {
    bundler_.Finalize(out);
  }

  void EncodeToken(std::string_view token, std::size_t role,
                   core::HyperVector<Dim, bool>* out) const {
    out->Clear();
    const auto [h1, h2] = detail::DoubleHash(token, seed_);
    for (int i = 0; i < k_; ++i) {
      const std::size_t pos =
          static_cast<std::size_t>((h1 + static_cast<std::uint64_t>(i) * h2) % Dim);
      out->SetBit(pos, true);
    }
    if (role != 0) {
      core::HyperVector<Dim, bool> rotated;
      rotated.Clear();
      core::PermuteRotate(*out, role, &rotated);
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

  void Reset() {
    bundler_.Reset();
    phase_ = 0;
  }

  void Update(std::size_t intensity) {
    const std::size_t clamped = (intensity > max_intensity_) ? max_intensity_ : intensity;
    core::HyperVector<Dim, bool> hv;
    hv.Clear();
    for (std::size_t i = 0; i < clamped && i < Dim; ++i) {
      const std::size_t index = order_[(phase_ + i) % Dim];
      hv.SetBit(index, true);
    }
    bundler_.Accumulate(hv);
    phase_ = (phase_ + clamped) % Dim;
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const {
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

  explicit SequentialNGramEncoder(std::uint64_t seed = 0x27d4eb2f165667c5ULL)
      : seed_(seed), head_(0), count_(0) {
    Reset();
  }

  void Reset() {
    bundler_.Reset();
    head_ = 0;
    count_ = 0;
  }

  void Update(std::uint64_t symbol) {
    history_[head_] = symbol;
    head_ = (head_ + 1) % Window;
    if (count_ < Window) {
      ++count_;
      return;  // wait until full window to form n-gram
    }

    core::HyperVector<Dim, bool> aggregate;
    aggregate.Clear();
    bool first = true;
    for (std::size_t i = 0; i < Window; ++i) {
      const std::size_t idx = (head_ + Window - 1 - i) % Window;
      core::HyperVector<Dim, bool> hv;
      hv.Clear();
      detail::GenerateRandomHypervector(seed_, history_[idx], &hv);
      if (i != 0) {
        core::HyperVector<Dim, bool> rotated;
        rotated.Clear();
        core::PermuteRotate(hv, i, &rotated);
        hv = rotated;
      }
      if (first) {
        aggregate = hv;
        first = false;
      } else {
        core::HyperVector<Dim, bool> bound;
        bound.Clear();
        core::Bind(aggregate, hv, &bound);
        aggregate = bound;
      }
    }
    bundler_.Accumulate(aggregate);
  }

  void Finalize(core::HyperVector<Dim, bool>* out) const {
    bundler_.Finalize(out);
  }

 private:
  std::uint64_t seed_;
  std::array<std::uint64_t, Window> history_{};
  std::size_t head_;
  std::size_t count_;
  core::BinaryBundler<Dim> bundler_;
};

}  // namespace encoding
}  // namespace hyperstream
