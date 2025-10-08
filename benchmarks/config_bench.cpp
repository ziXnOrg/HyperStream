// HyperStream Configuration Report Benchmark
// Prints active profile, defaults, CPU features, and selected backends.

#include <cstdio>
#include <cstdlib>
#include <exception>

#include "hyperstream/config.hpp"
#include "hyperstream/backend/capability.hpp"
#include "hyperstream/backend/policy.hpp"
#include "hyperstream/core/hypervector.hpp"

using hyperstream::core::HyperVector;

namespace {

template <std::size_t Dim>
void ReportSelectedBackends() {
  const auto rep = hyperstream::backend::Report<Dim>();
  std::printf("SelectedBackends/bind=%s,reason=\"%s\",hamming=%s,reason=\"%s\"\n",
              hyperstream::backend::GetBackendName(rep.bind_kind), rep.bind_reason,
              hyperstream::backend::GetBackendName(rep.hamming_kind), rep.hamming_reason);
}

void ReportFootprints() {
  using namespace hyperstream::config;
  const std::size_t d = kDefaultDimBits;
  // Representative configs
  const std::size_t hv_b = BinaryHyperVectorStorageBytes(d);
  const std::size_t cl_b = ClusterMemoryStorageBytes(d, 16);
  const std::size_t pm_b = PrototypeMemoryStorageBytes(d, 256);
  std::printf("Footprints/BinaryHV(dim=%zu)=%zub,ClusterMemory(dim=%zu,cap=16)=%zub,PrototypeMemory(dim=%zu,cap=256)=%zub\n",
              d, hv_b, d, cl_b, d, pm_b);
}

} // namespace

// Simple auto-tune microbench (disabled by default; enable with --auto-tune)
#include <chrono>
#include <cstring>

#if HS_X86_ARCH
template <std::size_t Dim>
static std::pair<double,double> MicrobenchHammingSSE2vsAVX2(std::size_t iters) {
  using namespace std::chrono;
  HyperVector<Dim, bool> a, b;
  a.Clear(); b.Clear();
  // Deterministic patterns
  for (std::size_t i = 0; i < Dim; i += 3) a.SetBit(i, true);
  for (std::size_t i = 1; i < Dim; i += 4) b.SetBit(i, true);
  volatile std::size_t sink = 0;
  // SSE2
  const auto t0 = high_resolution_clock::now();
  for (std::size_t i=0;i<iters;++i) sink += hyperstream::backend::sse2::HammingDistanceSSE2<Dim>(a,b);
  const auto t1 = high_resolution_clock::now();
  // AVX2 (if compiled)
  const auto t2 = high_resolution_clock::now();
#if defined(__AVX2__)
  for (std::size_t i=0;i<iters;++i) sink += hyperstream::backend::avx2::HammingDistanceAVX2<Dim>(a,b);
#endif
  const auto t3 = high_resolution_clock::now(); (void)sink;
  const double sse2_ms = duration<double,std::milli>(t1 - t0).count();
  const double avx2_ms = duration<double,std::milli>(t3 - t2).count();
  return {sse2_ms, avx2_ms};
}
#endif

int main(int argc, char** argv) try {
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);

  const bool auto_tune = (argc >= 2) && (std::strcmp(argv[1], "--auto-tune") == 0);

  // Profile and defaults
  std::printf("Config/profile=%s,default_dim_bits=%zu,default_capacity=%zu,heap_threshold_bytes=%zu\n",
              hyperstream::config::kActiveProfile,
              hyperstream::config::kDefaultDimBits,
              hyperstream::config::kDefaultCapacity,
              hyperstream::config::kHeapAllocThresholdBytes);

  // CPU features
  const std::uint32_t mask = hyperstream::backend::GetCpuFeatureMask();
  std::printf("CPUFeatures/mask=0x%08x,SSE2=%d,AVX2=%d\n", mask,
              hyperstream::backend::HasFeature(mask, hyperstream::backend::CpuFeature::SSE2) ? 1 : 0,
              hyperstream::backend::HasFeature(mask, hyperstream::backend::CpuFeature::AVX2) ? 1 : 0);

  // Threshold (env override aware)
  const std::size_t thr = hyperstream::backend::GetHammingThreshold();
  std::printf("Policy/HammingThreshold=%zu,overridden=%d\n", thr, hyperstream::backend::HammingThresholdOverridden() ? 1 : 0);

  // Selected backends and reasons for default dimension
  ReportSelectedBackends<hyperstream::config::kDefaultDimBits>();

  // Footprint estimates
  ReportFootprints();

#if HS_X86_ARCH
  if (auto_tune) {
    // Keep total runtime under ~2 seconds by limiting iterations per dimension.
    std::printf("AutoTune/Hamming begin\n");
    struct Case { std::size_t dim; std::size_t iters; } cases[] = {
      {8192,  8000}, {16384, 4000}, {32768, 2000}, {65536, 1000}
    };
    std::size_t recommended = 0;
    // Run per case with template instantiation by switch
    for (const auto& c : cases) {
      double sse2_ms=0, avx2_ms=0;
      switch (c.dim) {
        case 8192:  { auto p = MicrobenchHammingSSE2vsAVX2<8192>(c.iters);  sse2_ms=p.first; avx2_ms=p.second; break; }
        case 16384: { auto p = MicrobenchHammingSSE2vsAVX2<16384>(c.iters); sse2_ms=p.first; avx2_ms=p.second; break; }
        case 32768: { auto p = MicrobenchHammingSSE2vsAVX2<32768>(c.iters); sse2_ms=p.first; avx2_ms=p.second; break; }
        case 65536: { auto p = MicrobenchHammingSSE2vsAVX2<65536>(c.iters); sse2_ms=p.first; avx2_ms=p.second; break; }
      }
      const char* faster = (sse2_ms < avx2_ms) ? "sse2" : "avx2";
      std::printf("AutoTune/Hamming dim=%zu,sse2_ms=%.3f,avx2_ms=%.3f,faster=%s\n", c.dim, sse2_ms, avx2_ms, faster);
      if (!recommended && sse2_ms < avx2_ms) recommended = c.dim;
    }
    if (recommended) {
      std::printf("AutoTune/Hamming recommended_threshold=%zu (first dim where sse2 faster)\n", recommended);
    } else {
      std::printf("AutoTune/Hamming recommended_threshold=(none within tested range)\n");
    }
    std::printf("AutoTune/Hamming configured_threshold=%zu\n", thr);
  }
#else
  if (auto_tune) {
    std::printf("AutoTune/Hamming disabled on this architecture\n");
  }
#endif

  return EXIT_SUCCESS;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fprintf(stderr, "ERROR: unknown\n");
  return EXIT_FAILURE;
}

