#pragma once

// =============================================================================
// File:        include/hyperstream/core/ops.hpp
// Overview:    Core operations on HyperVector (binding, bundling, permutation,
//              similarity) with constexpr-friendly implementations.
// Mathematical Foundation: XOR bind for binary HVs; element-wise multiply/sum
//              for typed HVs; Hamming and cosine similarity functions provided.
// Security Considerations: No dynamic allocation; functions are noexcept where
//              safe; templates constrain types for Popcount and math helpers.
// Performance Considerations: Word-wise loops, constexpr/noexcept usage; avoid
//              unnecessary temporaries; mask excess bits in permutations.
// Examples:    See tests and encoding/* for usage in encoders and backends.
// =============================================================================
// Core operations over hypervectors: binding, bundling, permutation, similarity.
// Representation (HyperVector) is separate from operations to allow backend-specific
// optimizations while keeping a stable API. Header-only, constexpr-friendly, no deps.

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "hyperstream/core/hypervector.hpp"

namespace hyperstream::core {

// -----------------------------
// Internal helpers (no export)
// -----------------------------

namespace detail {
// Portable popcount. Kernighan's loop avoids CPU feature dependencies.
// constexpr and noexcept to enable compile-time evaluation and inlining.
// Constrained to unsigned integral types to prevent accidental misuse.
template <typename T,
          typename = std::enable_if_t<std::is_integral_v<T> && std::is_unsigned_v<T>>>
constexpr std::uint64_t Popcount64(T value) noexcept {
  std::uint64_t bit_count = 0;
  while (value) {
    value &= (value - 1);
    ++bit_count;
  }
  return bit_count;
}

// Inner-product term with conjugation for complex; supports arithmetic and complex types.
template <typename T>
constexpr double InnerProductTerm(const T& lhs, const T& rhs) noexcept {
  return static_cast<double>(lhs) * static_cast<double>(rhs);
}
template <typename R>
inline double InnerProductTerm(const std::complex<R>& lhs, const std::complex<R>& rhs) noexcept {
  return static_cast<double>((std::conj(lhs) * rhs).real());
}

// Squared norm |x|^2 for arithmetic and complex types.
template <typename T>
constexpr double SquaredNorm(const T& value) noexcept {
  return static_cast<double>(value) * static_cast<double>(value);
}
template <typename R>
inline double SquaredNorm(const std::complex<R>& value) noexcept {
  return static_cast<double>((std::conj(value) * value).real());
}
}  // namespace detail

// -----------------------------
// Binding
// -----------------------------

template <std::size_t Dim>
constexpr void Bind(const HyperVector<Dim, bool>& vector_a,
                    const HyperVector<Dim, bool>& vector_b,
                    HyperVector<Dim, bool>* out_vector) {
  // XOR binding for binary hypervectors. Involution property enables unbinding.
  for (std::size_t word_index = 0; word_index < HyperVector<Dim, bool>::WordCount(); ++word_index) {
    out_vector->Words()[word_index] =
        vector_a.Words()[word_index] ^ vector_b.Words()[word_index];
  }
}

template <std::size_t Dim, typename T>
constexpr void Bind(const HyperVector<Dim, T>& vector_a,
                    const HyperVector<Dim, T>& vector_b,
                    HyperVector<Dim, T>* out_vector) {
  // Element-wise multiplication for non-binary hypervectors.
  for (std::size_t element_index = 0; element_index < Dim; ++element_index) {
    (*out_vector)[element_index] = vector_a[element_index] * vector_b[element_index];
  }
}

// -----------------------------
// Bundling (superposition)
// -----------------------------

// Binary majority bundling: accumulate +/-1 per bit, then threshold once.
// Default counters are int16_t with saturation; optionally define HYPERSTREAM_BUNDLER_COUNTER_WIDE
// to use int32_t.

template <std::size_t Dim>
class BinaryBundler {
 public:
  // Rationale: use 16-bit signed saturating counters by default.
  // - Literature and hardware HDC accelerators commonly use narrow saturating counters (e.g., 5–8
  // bits)
  //   or binarized/rematerialized bundling to reduce memory (ETH/PULP-HD, DATE 2019–2021).
  // - int16 keeps the hot counter array smaller (≈2x smaller than int32), improving L1/L2 locality;
  //   SSE2/AVX2 provide efficient saturating operations for 16-bit lanes.
  // - ±32,767 votes per bit comfortably exceeds typical bundling needs (samples-per-class, short
  // windows). Compile-time option: define HYPERSTREAM_BUNDLER_COUNTER_WIDE to use 32-bit counters
  // and disable saturation. If an application requires >32k accumulations without decay, prefer
  // algorithmic changes (decay/chunk/binarized bundling); the wide-counter mode is an opt-in escape
  // hatch for niche cases.
#if defined(HYPERSTREAM_BUNDLER_COUNTER_WIDE)
  using counter_t = std::int32_t;
  static_assert(sizeof(counter_t) == 4, "Wide bundler counter expected 32-bit");
#else
  using counter_t = std::int16_t;
  static_assert(sizeof(counter_t) == 2, "Default bundler counter expected 16-bit");
#endif

  BinaryBundler() {
    Reset();
  }

  void Reset() {
    for (std::size_t i = 0; i < Dim; ++i) counters_[i] = 0;
  }

  void Accumulate(const HyperVector<Dim, bool>& hypervector) {
    // Streaming-friendly: avoid repeated thresholding to reduce drift.
#if defined(HYPERSTREAM_BUNDLER_COUNTER_WIDE)
    // Wide counters: simple add/sub (±2e9 capacity).
    for (std::size_t i = 0; i < Dim; ++i) {
      counters_[i] += hypervector.GetBit(i) ? 1 : -1;
    }
#else
    // Default: saturating add/sub to prevent overflow of 16-bit counters.
    for (std::size_t i = 0; i < Dim; ++i) {
      if (hypervector.GetBit(i)) {
        if (counters_[i] != std::numeric_limits<counter_t>::max()) ++counters_[i];
      } else {
        if (counters_[i] != std::numeric_limits<counter_t>::min()) --counters_[i];
      }
    }
#endif
  }

  void Finalize(HyperVector<Dim, bool>* out) const {
    for (std::size_t i = 0; i < Dim; ++i) {
      out->SetBit(i, counters_[i] >= 0);
    }
  }

 private:
  std::array<counter_t, Dim> counters_{};  // per-bit counters
};

// Numeric/complex bundling: element-wise sum; optionally normalized by caller.

template <std::size_t Dim, typename T>
constexpr void BundleAdd(const HyperVector<Dim, T>& vector_a,
                         const HyperVector<Dim, T>& vector_b,
                         HyperVector<Dim, T>* out_vector) {
  for (std::size_t element_index = 0; element_index < Dim; ++element_index) {
    (*out_vector)[element_index] = vector_a[element_index] + vector_b[element_index];
  }
}

// Optional compile-time helper: majority of two binary HVs (equivalent to OR).
// Provided to enable constexpr examples without changing BinaryBundler semantics.
template <std::size_t Dim>
constexpr void BundlePairMajority(const HyperVector<Dim, bool>& vector_a,
                                  const HyperVector<Dim, bool>& vector_b,
                                  HyperVector<Dim, bool>* out_vector) {
  for (std::size_t word_index = 0; word_index < HyperVector<Dim, bool>::WordCount(); ++word_index) {
    out_vector->Words()[word_index] =
        vector_a.Words()[word_index] | vector_b.Words()[word_index];
  }
}

// -----------------------------
// Permutation (position encoding)
// -----------------------------

// Define rotation as left-rotate by k positions.

template <std::size_t Dim>
constexpr void PermuteRotate(const HyperVector<Dim, bool>& input,
                             std::size_t rotate_by,
                             HyperVector<Dim, bool>* output) {
  // Optimized word-wise rotate with bit carry across 64-bit words.
  // Equivalent to left-rotate by k over Dim bits.
  const auto& input_words = input.Words();
  auto& output_words = output->Words();
  constexpr std::size_t word_count = HyperVector<Dim, bool>::WordCount();
  if constexpr (word_count == 0) {
    return;
  }
  const std::size_t rotate_words = (rotate_by / 64U) % word_count;  // whole-word rotation
  const std::size_t rotate_bits = rotate_by % 64U;                  // intra-word rotation

  if (rotate_bits == 0) {
    // Pure word rotation
    for (std::size_t i = 0; i < word_count; ++i) {
      output_words[i] = input_words[(i + word_count - rotate_words) % word_count];
    }
  } else {
    const auto shift_left = static_cast<unsigned>(rotate_bits);
    const unsigned shift_right = 64U - shift_left;
    for (std::size_t i = 0; i < word_count; ++i) {
      const std::size_t low_index = (i + word_count - rotate_words) % word_count;
      const std::size_t high_index = (i + word_count - rotate_words - 1) % word_count;
      const std::uint64_t low_word = input_words[low_index];
      const std::uint64_t high_word = input_words[high_index];
      output_words[i] = (low_word << shift_left) | (high_word >> shift_right);
    }
  }

  // Mask off any excess bits beyond Dim in the final word.
  constexpr std::size_t extra_bits = (HyperVector<Dim, bool>::WordCount() * 64ULL) - Dim;
  if constexpr (extra_bits > 0) {
    const std::uint64_t keep_mask = ~0ULL >> extra_bits;  // keep low (64-extra_bits) bits
    output_words[word_count - 1] &= keep_mask;
  }
}

template <std::size_t Dim, typename T>
constexpr void PermuteRotate(const HyperVector<Dim, T>& input,
                             std::size_t rotate_by,
                             HyperVector<Dim, T>* output) {
  const std::size_t shift = rotate_by % Dim;
  for (std::size_t i = 0; i < Dim; ++i) {
    const std::size_t src_index = (i + Dim - shift) % Dim;  // left-rotate by shift
    (*output)[i] = input[src_index];
  }
}

// -----------------------------
// Similarity
// -----------------------------

template <std::size_t Dim>
constexpr std::size_t HammingDistance(const HyperVector<Dim, bool>& vector_a,
                                      const HyperVector<Dim, bool>& vector_b) noexcept {
  std::size_t distance = 0;
  for (std::size_t word_index = 0; word_index < HyperVector<Dim, bool>::WordCount(); ++word_index) {
    distance += static_cast<std::size_t>(
        detail::Popcount64(vector_a.Words()[word_index] ^ vector_b.Words()[word_index]));
  }
  return distance;
}

template <std::size_t Dim>
struct NormalizedHammingArgs {
  const HyperVector<Dim, bool>* lhs;
  const HyperVector<Dim, bool>* rhs;
};

template <std::size_t Dim>
[[nodiscard]] inline float NormalizedHammingSimilarity(const NormalizedHammingArgs<Dim>& args) noexcept {
  // Map Hamming distance to [-1,1]: sim = 1 - 2*h/D. Clamp for numerical safety.
  const std::size_t h = HammingDistance(*args.lhs, *args.rhs);
  float sim = 1.0f - 2.0f * static_cast<float>(h) / static_cast<float>(Dim);
  if (sim > 1.0f) sim = 1.0f;
  if (sim < -1.0f) sim = -1.0f;
  return sim;
}

template <std::size_t Dim, typename T>
struct CosineArgs {
  const HyperVector<Dim, T>* lhs;
  const HyperVector<Dim, T>* rhs;
};

template <std::size_t Dim, typename T>
[[nodiscard]] inline float CosineSimilarity(const CosineArgs<Dim, T>& args) noexcept {
  // Cosine similarity; uses conjugation for complex elements.
  double num = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < Dim; ++i) {
    num += detail::InnerProductTerm((*args.lhs)[i], (*args.rhs)[i]);
    na += detail::SquaredNorm((*args.lhs)[i]);
    nb += detail::SquaredNorm((*args.rhs)[i]);
  }
  const double den = std::sqrt(na) * std::sqrt(nb) + 1e-12;  // epsilon to avoid div-by-zero
  return static_cast<float>(num / den);
}

}  // namespace hyperstream::core
