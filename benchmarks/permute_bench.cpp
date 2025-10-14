// HyperStream Permutation (Rotate) microbenchmark
// Measures throughput of core::PermuteRotate (bitwise) and a bench-local
// word-level rotate reference for binary HyperVectors.
// Output lines: name,dim_bits,bytes_per_iter,iters,secs,gb_per_sec

#include <chrono>
#include <cstdint>
#include <cstdio>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"

using hyperstream::core::HyperVector;
using hyperstream::core::PermuteRotate;

namespace {

template <std::size_t Dim>
static inline std::size_t bytes_per_iteration() {
  constexpr std::size_t words = HyperVector<Dim, bool>::WordCount();
  // Read input + write output
  return words * sizeof(std::uint64_t) * 2ULL;
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
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() <
           min_ms);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  if ((iters & 0xff) == 0) std::fprintf(stderr, "#sink=%llu\n", (unsigned long long)sink);
  return {iters, secs};
}

template <std::size_t Dim>
static void init_vectors(HyperVector<Dim, bool>* in) {
  in->Clear();
  for (std::size_t i = 0; i < Dim; i += 7) in->SetBit(i, true);
}

// Bench-local word-level left rotate by k bits across bit-packed storage.
// No bounds checks; intended for benchmarking only.
template <std::size_t Dim>
static inline void PermuteRotateWord_Ref(const HyperVector<Dim, bool>& in, std::size_t k,
                                         HyperVector<Dim, bool>* out) {
  const auto& iw = in.Words();
  auto& ow = out->Words();
  const std::size_t N = iw.size();
  const std::size_t q = (k / 64) % N;
  const std::size_t r = k % 64;
  if (r == 0) {
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
  // Mask off any excess bits beyond Dim in the final word
  constexpr std::size_t extra_bits = (HyperVector<Dim, bool>::WordCount() * 64ULL) - Dim;
  if constexpr (extra_bits > 0) {
    const std::uint64_t mask = ~0ULL >> extra_bits;  // keep low (64-extra_bits) bits
    ow[N - 1] &= mask;
  }
}

template <std::size_t Dim>
static void bench_one_dim() {
  constexpr std::size_t kRotate = 13;
  HyperVector<Dim, bool> in, out;
  init_vectors(&in);

  // Core bitwise rotate
  {
    const std::size_t kBytesPerIter = bytes_per_iteration<Dim>();
    auto [iters, secs] = run_for_ms(
        [&](volatile std::uint64_t* sink) {
          PermuteRotate(in, kRotate, &out);
          *sink ^= in.Words()[0];
        },
        300);
    const double gbps = (static_cast<double>(kBytesPerIter) * iters / secs) / 1e9;
    std::printf(
        "Permute/"
        "core_bitrotate,dim_bits=%zu,bytes_per_iter=%zu,iters=%zu,secs=%.6f,gb_per_sec=%.3f\n",
        Dim, kBytesPerIter, iters, secs, gbps);
  }

  // Word-level rotate reference
  {
    const std::size_t kBytesPerIter = bytes_per_iteration<Dim>();
    auto [iters, secs] = run_for_ms(
        [&](volatile std::uint64_t* sink) {
          PermuteRotateWord_Ref(in, kRotate, &out);
          *sink ^= in.Words()[0];
        },
        300);
    const double gbps = (static_cast<double>(kBytesPerIter) * iters / secs) / 1e9;
    std::printf(
        "Permute/"
        "word_rotate_ref,dim_bits=%zu,bytes_per_iter=%zu,iters=%zu,secs=%.6f,gb_per_sec=%.3f\n",
        Dim, kBytesPerIter, iters, secs, gbps);
  }
}

}  // namespace

int main() {
  bench_one_dim<1024>();
  bench_one_dim<2048>();
  bench_one_dim<4096>();
  bench_one_dim<8192>();
  bench_one_dim<10000>();
  bench_one_dim<16384>();
  bench_one_dim<65536>();
  bench_one_dim<262144>();
  bench_one_dim<1048576>();
  return 0;
}
