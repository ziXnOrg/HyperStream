// (moved helper functions below includes)

// (moved helper functions below includes)

// HyperStream Associative Memory microbenchmark
// Measures PrototypeMemory<Dim,Capacity>::Classify() throughput vs number of entries.
// Reports: name,dim_bits,capacity,size,iters,secs,queries_per_sec,eff_gb_per_sec

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <cstdlib>
#include <exception>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"
#include "hyperstream/backend/cpu_backend_sse2.hpp"
#include "hyperstream/backend/cpu_backend_avx2.hpp"
#include "hyperstream/config.hpp"
#include "hyperstream/backend/capability.hpp"
#include "hyperstream/backend/policy.hpp"

using hyperstream::core::HyperVector;
using hyperstream::memory::PrototypeMemory;

namespace {

// Simple splitmix generator for deterministic bit patterns
static inline std::uint64_t splitmix64(std::uint64_t& x) {
  x += 0x9e3779b97f4a7c15ULL;
  std::uint64_t z = x;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

template <std::size_t Dim>
static void fill_random(HyperVector<Dim, bool>* hv, std::uint64_t seed) {
  auto& words = hv->Words();
  for (std::size_t i = 0; i < words.size(); ++i) {
    words[i] = splitmix64(seed);
  }
  // Mask trailing bits beyond Dim
  constexpr std::size_t kWordBits = HyperVector<Dim, bool>::kWordBits;
  const std::size_t excess = words.size() * kWordBits - Dim;
  if (excess > 0) {
    const std::uint64_t mask = (excess == kWordBits) ? 0ULL : (~0ULL >> excess);
    words.back() &= mask;
  }
}

template <typename Fn>
static std::pair<std::size_t, double> run_for_ms(Fn&& fn, int min_ms) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::size_t iters = 0;
  volatile std::uint64_t sink = 0;
  do {
    fn(&sink);
    ++iters;
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() < min_ms);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  if ((iters & 0xff) == 0) std::fprintf(stderr, "#sink=%llu\n", (unsigned long long)sink);
  return {iters, secs};
}

template <std::size_t Dim, std::size_t Capacity, typename ClassifyFn>
static void bench_am(const char* name, std::size_t size, ClassifyFn&& classify) {
  PrototypeMemory<Dim, Capacity> am;
  // Fill entries deterministically up to 'size'
  std::uint64_t s = 12345;
  for (std::size_t i = 0; i < size && i < Capacity; ++i) {
    HyperVector<Dim, bool> hv; hv.Clear(); fill_random(&hv, s); (void)am.Learn(i + 1, hv);
  }
  HyperVector<Dim, bool> query; query.Clear(); fill_random(&query, s);

  const std::size_t words = HyperVector<Dim, bool>::WordCount();
  const std::size_t bytes_per_iter = (size * words + words) * sizeof(std::uint64_t); // approx

  auto [iters, secs] = run_for_ms([&](volatile std::uint64_t* sink){
    const auto lbl = classify(am, query);
    *sink ^= lbl;
  }, 300);

  const double qps = static_cast<double>(iters) / secs;
  const double eff_gbps = (static_cast<double>(bytes_per_iter) * iters / secs) / 1e9;
  std::printf("%s,dim_bits=%zu,capacity=%zu,size=%zu,iters=%zu,secs=%.6f,queries_per_sec=%.1f,eff_gb_per_sec=%.3f\n",
              name, Dim, Capacity, size, iters, secs, qps, eff_gbps);
}

template <std::size_t Dim>
static void run_one_dim() {
  // Choose representative capacities/sizes
  bench_am<Dim, 256>("AM/core", 256, [](auto& am, const auto& q){ return am.Classify(q, 0); });
  bench_am<Dim, 1024>("AM/core", 1024, [](auto& am, const auto& q){ return am.Classify(q, 0); });

  // SSE2 distance functor
  bench_am<Dim, 1024>("AM/sse2", 1024, [](const auto& am, const auto& q){
    return am.Classify(q, [](const auto& a, const auto& b){
      return hyperstream::backend::sse2::HammingDistanceSSE2<Dim>(a, b);
    }, 0);
  });

  // AVX2 distance functor
#if defined(__AVX2__)
  bench_am<Dim, 1024>("AM/avx2", 1024, [](const auto& am, const auto& q){
    return am.Classify(q, [](const auto& a, const auto& b){
      return hyperstream::backend::avx2::HammingDistanceAVX2<Dim>(a, b);
    }, 0);
  });
#endif
}

} // namespace

int main(int argc, char** argv) try {
  // Suppress unused warnings in some toolchains
  (void)argc; (void)argv;
  // Make stdout/stderr unbuffered to ensure visibility across environments
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);

  // Print active configuration
  std::printf("Config/profile=%s,default_dim_bits=%zu,default_capacity=%zu\n",
              hyperstream::config::kActiveProfile,
              hyperstream::config::kDefaultDimBits,
              hyperstream::config::kDefaultCapacity);

  // Use a typical binary dimension for HDC
  run_one_dim<10000>();
  run_one_dim<16384>();
  run_one_dim<65536>();

  return EXIT_SUCCESS;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fprintf(stderr, "ERROR: unknown\n");
  return EXIT_FAILURE;
}
