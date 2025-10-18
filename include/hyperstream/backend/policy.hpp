#pragma once

// Backend selection policy: choose optimal implementations based on runtime CPU
// feature detection, with constexpr/compile-time fallbacks. Adds simple
// dimension-based heuristics informed by host benchmarks.

#include <cstddef>
#include <cstdint>
#include <cstdlib>  // getenv

#include "hyperstream/backend/capability.hpp"
#include "hyperstream/config.hpp"
#include "hyperstream/core/ops.hpp"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#include "hyperstream/backend/cpu_backend_avx2.hpp"
#include "hyperstream/backend/cpu_backend_sse2.hpp"
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
#include "hyperstream/backend/cpu_backend_neon.hpp"
#endif

namespace hyperstream::backend {

// Heuristic thresholds (tunable). On this host, SSE2 Hamming tends to win at large dims.
// Users can override this at runtime via the environment variable
// HYPERSTREAM_HAMMING_SSE2_THRESHOLD. If unset or invalid, the default is used.
static constexpr std::size_t kHammingPreferSSE2DimThreshold = 16384;

/// Returns the Hamming SSE2-preference threshold. If the environment variable
/// HYPERSTREAM_HAMMING_SSE2_THRESHOLD is set to a positive integer, it is used.
/// Otherwise, returns kHammingPreferSSE2DimThreshold.
[[nodiscard]] inline std::size_t GetHammingThreshold() noexcept {
  const char* env = std::getenv("HYPERSTREAM_HAMMING_SSE2_THRESHOLD");
  if (env == nullptr || *env == '\0') {
    return kHammingPreferSSE2DimThreshold;
  }
  char* end = nullptr;
  static constexpr int kBase10 = 10;
  const auto parsed_value = std::strtoull(env, &end, kBase10);
  if (end == env || *end != '\0') {
    return kHammingPreferSSE2DimThreshold;  // not a pure number
  }
  if (parsed_value == 0ULL) {
    return kHammingPreferSSE2DimThreshold;
  }
  return static_cast<std::size_t>(parsed_value);
}

/// True if the HYPERSTREAM_HAMMING_SSE2_THRESHOLD environment variable is set to a valid value.
[[nodiscard]] inline bool HammingThresholdOverridden() noexcept {
  const char* env = std::getenv("HYPERSTREAM_HAMMING_SSE2_THRESHOLD");
  if (env == nullptr || *env == '\0') {
    return false;
  }
  char* end = nullptr;
  static constexpr int kBase10 = 10;
  (void)std::strtoull(env, &end, kBase10);
  return (end != env && *end == '\0');
}

/** Kind of backend selected by the policy. */
enum class BackendKind : std::uint8_t { Scalar = 0, SSE2 = 1, AVX2 = 2, NEON = 3 };

/** Returns a human-readable backend name: "scalar", "sse2", or "avx2". */
[[nodiscard]] inline const char* GetBackendName(BackendKind backend_kind) noexcept {
  switch (backend_kind) {
    case BackendKind::Scalar:
      return "scalar";
    case BackendKind::SSE2:
      return "sse2";
    case BackendKind::AVX2:
      return "avx2";
    case BackendKind::NEON:
      return "neon";
    default:
      return "unknown";
  }
}

// Function pointer types
template <std::size_t Dim>
using BindFn = void (*)(const core::HyperVector<Dim, bool>&, const core::HyperVector<Dim, bool>&,
                        core::HyperVector<Dim, bool>*);

template <std::size_t Dim>
using HammingFn = std::size_t (*)(const core::HyperVector<Dim, bool>&,
                                  const core::HyperVector<Dim, bool>&);

// Decision helpers
namespace detail {
struct DecisionInputs {
  std::size_t dim_bits;
  std::uint32_t feature_mask;
};
struct Decision {
  BackendKind kind;
  const char* reason;
};

inline Decision DecideBind(DecisionInputs inputs) noexcept {
  (void)inputs.dim_bits;  // not used in current bind heuristic
#if defined(HYPERSTREAM_FORCE_SCALAR)
  (void)inputs.feature_mask;
  return {BackendKind::Scalar, "forced scalar"};
#else
  if (HasFeature(inputs.feature_mask, CpuFeature::AVX2)) {
    return {BackendKind::AVX2, "wider vectors (256b)"};
  }
  if (HasFeature(inputs.feature_mask, CpuFeature::SSE2)) {
    return {BackendKind::SSE2, "SSE2 available"};
  }
  if (HasFeature(inputs.feature_mask, CpuFeature::NEON)) {
    return {BackendKind::NEON, "NEON available"};
  }
  return {BackendKind::Scalar, "no SIMD detected"};
#endif
}

inline Decision DecideHamming(DecisionInputs inputs) noexcept {
  // Silence MSVC C4100 in compilation modes where dim/mask are not used
  (void)inputs.dim_bits;
  (void)inputs.feature_mask;
#if defined(HYPERSTREAM_FORCE_SCALAR)
  return {BackendKind::Scalar, "forced scalar"};
#else
  if (HasFeature(inputs.feature_mask, CpuFeature::AVX2)) {
    const std::size_t thr = GetHammingThreshold();
    if (inputs.dim_bits >= thr && HasFeature(inputs.feature_mask, CpuFeature::SSE2)) {
      return {BackendKind::SSE2, "preferred for large dims (threshold heuristic)"};
    }
    return {BackendKind::AVX2, "wider vectors (256b)"};
  }
  if (HasFeature(inputs.feature_mask, CpuFeature::SSE2)) {
    return {BackendKind::SSE2, "SSE2 available"};
  }
  if (HasFeature(inputs.feature_mask, CpuFeature::NEON)) {
    return {BackendKind::NEON, "NEON available"};
  }
  return {BackendKind::Scalar, "no SIMD detected"};
#endif
}
}  // namespace detail

// Compile-time override mapping (placeholder for future string mapping)
namespace detail {
constexpr int BackendOverrideTag() {
#if defined(HYPERSTREAM_FORCE_SCALAR)
  return 0;  // scalar only
#elif defined(HYPERSTREAM_BACKEND_OVERRIDE)
  return -1;  // unknown/handle at runtime
#else
  return -1;  // no override
#endif
}
}  // namespace detail

// Select Bind implementation using decision helper
template <std::size_t Dim>
[[nodiscard]] inline BindFn<Dim> SelectBindBackend(
    std::uint32_t feature_mask = GetCpuFeatureMask()) noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  const auto d = detail::DecideBind(detail::DecisionInputs{Dim, feature_mask});
  switch (d.kind) {
    case BackendKind::AVX2:
      return &avx2::BindAVX2<Dim>;
    case BackendKind::SSE2:
      return &sse2::BindSSE2<Dim>;
    default:
      return &core::Bind<Dim>;
  }
#elif defined(__aarch64__) || defined(_M_ARM64)
  (void)feature_mask;  // On ARM64, ignore synthetic x86 bits; use host features
  const auto d = detail::DecideBind(detail::DecisionInputs{Dim, GetCpuFeatureMask()});
  switch (d.kind) {
    case BackendKind::NEON:
      return &neon::BindNEON<Dim>;
    default:
      return &core::Bind<Dim>;
  }
#else
  (void)feature_mask;  // other arch: scalar
  return &core::Bind<Dim>;
#endif
}

// Select Hamming distance implementation using decision helper
template <std::size_t Dim>
[[nodiscard]] inline HammingFn<Dim> SelectHammingBackend(
    std::uint32_t feature_mask = GetCpuFeatureMask()) noexcept {
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
  const auto d = detail::DecideHamming(detail::DecisionInputs{Dim, feature_mask});
  switch (d.kind) {
    case BackendKind::AVX2:
      return &avx2::HammingDistanceAVX2<Dim>;
    case BackendKind::SSE2:
      return &sse2::HammingDistanceSSE2<Dim>;
    default:
      return &core::HammingDistance<Dim>;
  }
#elif defined(__aarch64__) || defined(_M_ARM64)
  (void)feature_mask;  // On ARM64, ignore synthetic x86 bits; use host features
  const auto d = detail::DecideHamming(detail::DecisionInputs{Dim, GetCpuFeatureMask()});
  switch (d.kind) {
    case BackendKind::NEON:
      return &neon::HammingDistanceNEON<Dim>;
    default:
      return &core::HammingDistance<Dim>;
  }
#else
  (void)feature_mask;  // other arch: scalar
  return &core::HammingDistance<Dim>;
#endif
}

// Policy report
/** Summary of policy decisions for a given dimension and CPU feature mask. */
struct PolicyReport {
  std::size_t dim_bits;        ///< Dimension in bits
  std::uint32_t feature_mask;  ///< CPU feature bitmask
  BackendKind bind_kind;
  const char* bind_reason;  ///< Bind backend + rationale
  BackendKind hamming_kind;
  const char* hamming_reason;  ///< Hamming backend + rationale
};

/** Reports backend selections and reasons for Dim and optional feature_mask. */
template <std::size_t Dim>
[[nodiscard]] inline PolicyReport Report(
    std::uint32_t feature_mask = GetCpuFeatureMask()) noexcept {
  const auto b = detail::DecideBind(detail::DecisionInputs{Dim, feature_mask});
  const auto h = detail::DecideHamming(detail::DecisionInputs{Dim, feature_mask});
  return PolicyReport{Dim, feature_mask, b.kind, b.reason, h.kind, h.reason};
}

}  // namespace hyperstream::backend
