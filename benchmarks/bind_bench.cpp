// HyperStream Bind microbenchmark (no external deps)
// Measures throughput of core::Bind (scalar) vs available SIMD backends (SSE2/AVX2)
// across a set of fixed hypervector dimensions.
// Output: CSV-like lines per benchmark: name,dimension_bits,bytes_per_iter,iterations,seconds,gb_per_sec

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include "hyperstream/backend/policy.hpp"
#if HS_X86_ARCH
#include <immintrin.h>
#include "hyperstream/backend/cpu_backend_sse2.hpp"
#include "hyperstream/backend/cpu_backend_avx2.hpp"
#endif
#if HS_ARM64_ARCH
#include "hyperstream/backend/cpu_backend_neon.hpp"
#endif
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

using hyperstream::core::HyperVector;
using hyperstream::core::Bind;
#if HS_X86_ARCH
using hyperstream::backend::sse2::BindSSE2;
using hyperstream::backend::avx2::BindAVX2;
#endif

namespace {

template <std::size_t Dim>
static inline std::size_t bytes_per_iteration() {
  constexpr std::size_t words = HyperVector<Dim, bool>::WordCount();
  // Read a + read b + write out
  return words * sizeof(std::uint64_t) * 3ULL;
}

#if defined(__AVX2__)
// Inline AVX2 reference (loadu/xor/storeu) for A/B without alignment or NT logic.
template <std::size_t Dim>
static inline void BindAVX2_Ref(const HyperVector<Dim, bool>& a,
                                const HyperVector<Dim, bool>& b,
                                HyperVector<Dim, bool>* out) {
  const auto& aw = a.Words();
  const auto& bw = b.Words();
  auto& ow = out->Words();
  const std::size_t num = aw.size();
  const std::size_t avx_words = (num / 4) * 4;
  std::size_t i = 0;
  for (; i < avx_words; i += 4) {
    __m256i va = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&aw[i]));
    __m256i vb = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&bw[i]));
    __m256i vx = _mm256_xor_si256(va, vb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(&ow[i]), vx);
  }
  for (; i < num; ++i) ow[i] = aw[i] ^ bw[i];
}
#endif

// Basic steady-state loop running for at least min_ms milliseconds.
// Returns pair {iterations, seconds}.
template <typename Fn>
static std::pair<std::size_t, double> run_for_ms(Fn&& fn, int min_ms) {
  using clock = std::chrono::steady_clock;
  const auto t0 = clock::now();
  std::size_t iters = 0;
  volatile std::uint64_t sink = 0; // prevent over-optimization
  do {
    fn(&sink);
    ++iters;
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() < min_ms);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  // Print sink occasionally to avoid optimizing everything away (still minimal overhead)
  if ((iters & 0xff) == 0) std::fprintf(stderr, "#sink=%llu\n", (unsigned long long)sink);
  return {iters, secs};
}

// Benchmark a single implementation identified by name for a fixed dimension.
// The callable do_bind(out_sink*) must perform one Bind over initialized a,b,out.
template <std::size_t Dim, typename DoBind>
static void bench_impl(const char* name, DoBind&& do_bind) {
  const std::size_t kBytesPerIter = bytes_per_iteration<Dim>();
  auto [iters, secs] = run_for_ms(do_bind, 300); // ~300ms per case
  const double total_bytes = static_cast<double>(kBytesPerIter) * static_cast<double>(iters);
  const double gbps = (total_bytes / secs) / 1e9; // GB/s
  std::printf("%s,dim_bits=%zu,bytes_per_iter=%zu,iters=%zu,secs=%.6f,gb_per_sec=%.3f\n",
              name, Dim, kBytesPerIter, iters, secs, gbps);
}

// Helper to initialize test vectors deterministically.
template <std::size_t Dim>
static void init_vectors(HyperVector<Dim, bool>* a, HyperVector<Dim, bool>* b) {
  a->Clear();
  b->Clear();
  for (std::size_t i = 0; i < Dim; i += 3) a->SetBit(i, true);
  for (std::size_t i = 1; i < Dim; i += 5) b->SetBit(i, true);
}

// Bench a single dimension for scalar and available SIMD backends.
template <std::size_t D>
static void run_one() {
  HyperVector<D, bool> a, b, out;
  init_vectors(&a, &b);

  // A/B-1 ordering: AVX2 first, then core, then SSE2. Also add AVX2_Ref for A/B-2.
  #if defined(__AVX2__)
  bench_impl<D>("Bind/avx2", [&](volatile std::uint64_t* sink) {
    BindAVX2(a, b, &out);
    *sink ^= a.Words()[0];
  });
  bench_impl<D>("Bind/avx2_ref", [&](volatile std::uint64_t* sink) {
    BindAVX2_Ref(a, b, &out);
    *sink ^= a.Words()[0];
  });
  #endif

  bench_impl<D>("Bind/core", [&](volatile std::uint64_t* sink) {
    Bind(a, b, &out);
    *sink ^= a.Words()[0];
  });

#if HS_X86_ARCH
  bench_impl<D>("Bind/sse2", [&](volatile std::uint64_t* sink) {
    BindSSE2(a, b, &out);
    *sink ^= a.Words()[0];
  });
#endif
#if HS_ARM64_ARCH
  bench_impl<D>("Bind/neon", [&](volatile std::uint64_t* sink) {
    hyperstream::backend::neon::BindNEON(a, b, &out);
    *sink ^= a.Words()[0];
  });
#endif

}

} // namespace

int main() {
  // Representative sizes (bits)
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
