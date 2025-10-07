#pragma once

// Core operations over hypervectors: binding, bundling, permutation, similarity.
// Representation (HyperVector) is separate from operations to allow backend-specific
// optimizations while keeping a stable API. Header-only, constexpr-friendly, no deps.

#include <array>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <complex>
#include <cmath>
#include <limits>
#include "hyperstream/core/hypervector.hpp"

namespace hyperstream {
namespace core {

 // -----------------------------
 // Internal helpers (no export)
 // -----------------------------
 
namespace detail {
  // Portable popcount. Kernighan's loop avoids CPU feature dependencies.
  // constexpr and noexcept to enable compile-time evaluation and inlining.
  // Constrained to unsigned integral types to prevent accidental misuse.
  template <typename T, typename = std::enable_if_t<std::is_integral<T>::value && std::is_unsigned<T>::value>>
  constexpr inline std::uint64_t Popcount64(T x) noexcept {
    std::uint64_t c = 0;
    while (x) {
      x &= (x - 1);
      ++c;
    }
    return c;
  }

// Inner-product term with conjugation for complex; supports arithmetic and complex types.
template <typename T>
constexpr double InnerProductTerm(const T& a, const T& b) noexcept {
  return static_cast<double>(a) * static_cast<double>(b);
}
template <typename R>
inline double InnerProductTerm(const std::complex<R>& a, const std::complex<R>& b) noexcept {
  return static_cast<double>((std::conj(a) * b).real());
}

// Squared norm |x|^2 for arithmetic and complex types.
template <typename T>
constexpr double SquaredNorm(const T& x) noexcept {
  return static_cast<double>(x) * static_cast<double>(x);
}
template <typename R>
inline double SquaredNorm(const std::complex<R>& x) noexcept {
  return static_cast<double>((std::conj(x) * x).real());
}
}  // namespace detail

// -----------------------------
// Binding
// -----------------------------

template <std::size_t Dim>
inline void Bind(const HyperVector<Dim, bool>& a,
                 const HyperVector<Dim, bool>& b,
                 HyperVector<Dim, bool>* out) {
  // XOR binding for binary hypervectors. Involution property enables unbinding.
  for (std::size_t w = 0; w < HyperVector<Dim, bool>::WordCount(); ++w) {
    out->Words()[w] = a.Words()[w] ^ b.Words()[w];
  }
}

template <std::size_t Dim, typename T>
inline void Bind(const HyperVector<Dim, T>& a,
                 const HyperVector<Dim, T>& b,
                 HyperVector<Dim, T>* out) {
  // Element-wise multiplication for non-binary hypervectors.
  for (std::size_t i = 0; i < Dim; ++i) {
    (*out)[i] = a[i] * b[i];
  }
}

// -----------------------------
// Bundling (superposition)
// -----------------------------

// Binary majority bundling: accumulate +/-1 per bit, then threshold once.
// Default counters are int16_t with saturation; optionally define HYPERSTREAM_BUNDLER_COUNTER_WIDE to use int32_t.

template <std::size_t Dim>
class BinaryBundler {
 public:
  // Rationale: use 16-bit signed saturating counters by default.
  // - Literature and hardware HDC accelerators commonly use narrow saturating counters (e.g., 5–8 bits)
  //   or binarized/rematerialized bundling to reduce memory (ETH/PULP-HD, DATE 2019–2021).
  // - int16 keeps the hot counter array smaller (≈2x smaller than int32), improving L1/L2 locality;
  //   SSE2/AVX2 provide efficient saturating operations for 16-bit lanes.
  // - ±32,767 votes per bit comfortably exceeds typical bundling needs (samples-per-class, short windows).
  // Compile-time option: define HYPERSTREAM_BUNDLER_COUNTER_WIDE to use 32-bit counters and disable saturation.
  // If an application requires >32k accumulations without decay, prefer algorithmic changes (decay/chunk/binarized
  // bundling); the wide-counter mode is an opt-in escape hatch for niche cases.
#if defined(HYPERSTREAM_BUNDLER_COUNTER_WIDE)
  using counter_t = std::int32_t;
  static_assert(sizeof(counter_t) == 4, "Wide bundler counter expected 32-bit");
#else
  using counter_t = std::int16_t;
  static_assert(sizeof(counter_t) == 2, "Default bundler counter expected 16-bit");
#endif

  BinaryBundler() { Reset(); }

  void Reset() {
    for (std::size_t i = 0; i < Dim; ++i) counters_[i] = 0;
  }

  void Accumulate(const HyperVector<Dim, bool>& hv) {
    // Streaming-friendly: avoid repeated thresholding to reduce drift.
#if defined(HYPERSTREAM_BUNDLER_COUNTER_WIDE)
    // Wide counters: simple add/sub (±2e9 capacity).
    for (std::size_t i = 0; i < Dim; ++i) {
      counters_[i] += hv.GetBit(i) ? 1 : -1;
    }
#else
    // Default: saturating add/sub to prevent overflow of 16-bit counters.
    for (std::size_t i = 0; i < Dim; ++i) {
      if (hv.GetBit(i)) {
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
inline void BundleAdd(const HyperVector<Dim, T>& a,
                      const HyperVector<Dim, T>& b,
                      HyperVector<Dim, T>* out) {
  for (std::size_t i = 0; i < Dim; ++i) {
    (*out)[i] = a[i] + b[i];
  }
}

// -----------------------------
// Permutation (position encoding)
// -----------------------------

// Define rotation as left-rotate by k positions.

template <std::size_t Dim>
inline void PermuteRotate(const HyperVector<Dim, bool>& in,
                          std::size_t k,
                          HyperVector<Dim, bool>* out) {
  // Optimized word-wise rotate with bit carry across 64-bit words.
  // Equivalent to left-rotate by k over Dim bits.
  const auto& iw = in.Words();
  auto& ow = out->Words();
  constexpr std::size_t N = HyperVector<Dim, bool>::WordCount();
  if constexpr (N == 0) {
    return;
  }
  const std::size_t q = (k / 64) % N;      // whole-word rotation
  const std::size_t r = k % 64;            // intra-word rotation

  if (r == 0) {
    // Pure word rotation
    for (std::size_t i = 0; i < N; ++i) {
      ow[i] = iw[(i + N - q) % N];
    }
  } else {
    const unsigned s = static_cast<unsigned>(r);
    const unsigned t = 64u - s;
    for (std::size_t i = 0; i < N; ++i) {
      const std::size_t lo_idx = (i + N - q) % N;
      const std::size_t hi_idx = (i + N - q - 1) % N;
      const std::uint64_t lo = iw[lo_idx];
      const std::uint64_t hi = iw[hi_idx];
      ow[i] = (lo << s) | (hi >> t);
    }
  }

  // Mask off any excess bits beyond Dim in the final word.
  constexpr std::size_t extra_bits = (HyperVector<Dim, bool>::WordCount() * 64ULL) - Dim;
  if constexpr (extra_bits > 0) {
    const std::uint64_t mask = ~0ULL >> extra_bits; // keep low (64-extra_bits) bits
    ow[N - 1] &= mask;
  }
}

template <std::size_t Dim, typename T>
inline void PermuteRotate(const HyperVector<Dim, T>& in,
                          std::size_t k,
                          HyperVector<Dim, T>* out) {
  const std::size_t s = k % Dim;
  for (std::size_t i = 0; i < Dim; ++i) {
    const std::size_t src = (i + Dim - s) % Dim;  // left-rotate by s
    (*out)[i] = in[src];
  }
}

// -----------------------------
// Similarity
// -----------------------------

template <std::size_t Dim>
inline std::size_t HammingDistance(const HyperVector<Dim, bool>& a,
                                   const HyperVector<Dim, bool>& b) {
  std::size_t dist = 0;
  for (std::size_t w = 0; w < HyperVector<Dim, bool>::WordCount(); ++w) {
    dist += static_cast<std::size_t>(detail::Popcount64(a.Words()[w] ^ b.Words()[w]));
  }
  return dist;
}

template <std::size_t Dim>
inline float NormalizedHammingSimilarity(const HyperVector<Dim, bool>& a,
                                         const HyperVector<Dim, bool>& b) {
  // Map Hamming distance to [-1,1]: sim = 1 - 2*h/D. Clamp for numerical safety.
  const std::size_t h = HammingDistance(a, b);
  float sim = 1.0f - 2.0f * static_cast<float>(h) / static_cast<float>(Dim);
  if (sim > 1.0f) sim = 1.0f;
  if (sim < -1.0f) sim = -1.0f;
  return sim;
}

template <std::size_t Dim, typename T>
inline float CosineSimilarity(const HyperVector<Dim, T>& a,
                              const HyperVector<Dim, T>& b) {
  // Cosine similarity; uses conjugation for complex elements.
  double num = 0.0;
  double na = 0.0;
  double nb = 0.0;
  for (std::size_t i = 0; i < Dim; ++i) {
    num += detail::InnerProductTerm(a[i], b[i]);
    na += detail::SquaredNorm(a[i]);
    nb += detail::SquaredNorm(b[i]);
  }
  const double den = std::sqrt(na) * std::sqrt(nb) + 1e-12;  // epsilon to avoid div-by-zero
  return static_cast<float>(num / den);
}

}  // namespace core
}  // namespace hyperstream
