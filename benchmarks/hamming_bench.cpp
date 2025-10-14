// HyperStream Hamming distance microbenchmark (no external deps)
// Measures throughput of Hamming distance for binary HyperVectors:
//   core::HammingDistance (scalar), SSE2, and AVX2 (Harley–Seal)
// Output: CSV-like: name,dimension_bits,bytes_per_iter,iterations,seconds,gb_per_sec

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

#include "hyperstream/backend/policy.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#if HS_X86_ARCH
#include "hyperstream/backend/cpu_backend_avx2.hpp"
#include "hyperstream/backend/cpu_backend_sse2.hpp"
#endif
#if HS_ARM64_ARCH
#include "hyperstream/backend/cpu_backend_neon.hpp"
#endif

using hyperstream::core::HammingDistance;
using hyperstream::core::HyperVector;
#if HS_X86_ARCH
using hyperstream::backend::avx2::HammingDistanceAVX2;
using hyperstream::backend::sse2::HammingDistanceSSE2;
#endif

namespace {

template <std::size_t Dim>
static inline std::size_t bytes_per_iteration() {
  constexpr std::size_t words = HyperVector<Dim, bool>::WordCount();
  // Read a + read b (no write)
  return words * sizeof(std::uint64_t) * 2ULL;
}

template <typename Fn>
static std::pair<std::size_t, double> run_for_ms(Fn&& fn, int min_ms) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::size_t iters = 0;
  volatile std::size_t sink = 0;
  do {
    fn(&sink);
    ++iters;
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() <
           min_ms);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  if ((iters & 0xff) == 0) std::fprintf(stderr, "#sink=%zu\n", (size_t)sink);
  return {iters, secs};
}

template <std::size_t Dim, typename DoDist>
static void bench_impl(const char* name, DoDist&& do_dist) {
  const std::size_t kBytesPerIter = bytes_per_iteration<Dim>();
  auto [iters, secs] = run_for_ms(do_dist, 300);
  const double total_bytes = static_cast<double>(kBytesPerIter) * static_cast<double>(iters);
  const double gbps = (total_bytes / secs) / 1e9;
  std::printf("%s,dim_bits=%zu,bytes_per_iter=%zu,iters=%zu,secs=%.6f,gb_per_sec=%.3f\n", name, Dim,
              kBytesPerIter, iters, secs, gbps);
}

template <std::size_t Dim>
static void init_vectors(HyperVector<Dim, bool>* a, HyperVector<Dim, bool>* b) {
  a->Clear();
  b->Clear();
  for (std::size_t i = 0; i < Dim; i += 3) a->SetBit(i, true);
  for (std::size_t i = 1; i < Dim; i += 5) b->SetBit(i, true);
}

template <std::size_t D>
static void run_one() {
  HyperVector<D, bool> a, b;
  init_vectors(&a, &b);

  bench_impl<D>("Hamming/core",
                [&](volatile std::size_t* sink) { *sink ^= HammingDistance(a, b); });

#if HS_X86_ARCH
  bench_impl<D>("Hamming/sse2",
                [&](volatile std::size_t* sink) { *sink ^= HammingDistanceSSE2(a, b); });
#endif

#if defined(__AVX2__)
  bench_impl<D>("Hamming/avx2",
                [&](volatile std::size_t* sink) { *sink ^= HammingDistanceAVX2(a, b); });
#endif

#if HS_ARM64_ARCH
  bench_impl<D>("Hamming/neon", [&](volatile std::size_t* sink) {
    *sink ^= hyperstream::backend::neon::HammingDistanceNEON<D>(a, b);
  });
#endif
}

}  // namespace

int main() {
  run_one<1024>();
  run_one<2048>();
  run_one<4096>();
  run_one<8192>();
  run_one<10000>();
  run_one<16384>();
  run_one<65536>();
  run_one<262144>();
  run_one<1048576>();
  return 0;
}
