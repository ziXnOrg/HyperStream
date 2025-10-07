// HyperStream Cluster Memory microbenchmark
// Measures ClusterMemory<Dim,Capacity>::Update() and Finalize() throughput.
// Reports: name,dim_bits,capacity,updates,iters,secs,updates_per_sec,finalizes_per_sec

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"
#include "hyperstream/config.hpp"
#include "hyperstream/backend/capability.hpp"
#include "hyperstream/backend/policy.hpp"

using hyperstream::core::HyperVector;
using hyperstream::memory::ClusterMemory;

namespace {

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
  constexpr std::size_t WB = HyperVector<Dim, bool>::kWordBits;
  const std::size_t excess = words.size() * WB - Dim;
  if (excess > 0) {
    const std::uint64_t mask = (excess == WB) ? 0ULL : (~0ULL >> excess);
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

template <std::size_t Dim, std::size_t Capacity>
static void bench_cluster(const char* name, std::size_t updates) {
  ClusterMemory<Dim, Capacity> cmem;
  std::uint64_t s = 1;

  // Measure Update throughput
  auto [iters_u, secs_u] = run_for_ms([&](volatile std::uint64_t* sink){
    for (std::size_t i = 0; i < updates; ++i) {
      HyperVector<Dim, bool> hv; hv.Clear(); fill_random(&hv, s);
      (void)cmem.Update(42, hv);
      *sink ^= hv.Words()[0];
    }
  }, 150);

  // Emit an early metrics line for the update phase so users see output quickly
  const double updates_ps_early = static_cast<double>(iters_u) * updates / secs_u;
  std::printf("%s-update,dim_bits=%zu,capacity=%zu,updates=%zu,update_iters=%zu,update_secs=%.6f,updates_per_sec=%.1f\n",
              name, Dim, Capacity, updates, iters_u, secs_u, updates_ps_early);
  std::fflush(stdout);

  // Measure Finalize throughput
  HyperVector<Dim, bool> out;
  auto [iters_f, secs_f] = run_for_ms([&](volatile std::uint64_t* sink){
    cmem.Finalize(42, &out);
    *sink ^= out.Words()[0];
  }, 150);

  const double updates_ps = static_cast<double>(iters_u) * updates / secs_u;
  const double finalizes_ps = static_cast<double>(iters_f) / secs_f;
  std::printf("%s,dim_bits=%zu,capacity=%zu,updates=%zu,update_iters=%zu,update_secs=%.6f,updates_per_sec=%.1f,finalize_iters=%zu,finalize_secs=%.6f,finalizes_per_sec=%.1f\n",
              name, Dim, Capacity, updates, iters_u, secs_u, updates_ps, iters_f, secs_f, finalizes_ps);
  std::fflush(stdout);
}

template <std::size_t Dim>
static void run_one_dim() {
  bench_cluster<Dim, 16>("Cluster/update_finalize", 100);
}

} // namespace

int main(int argc, char** argv) try {
  // Suppress unused warnings in some toolchains
  (void)argc; (void)argv;
  // Make stdout/stderr unbuffered to ensure visibility across environments
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);

  if (argc == 1) {
    // Print active configuration
    std::printf("Config/profile=%s,default_dim_bits=%zu,default_capacity=%zu\n",
                hyperstream::config::kActiveProfile,
                hyperstream::config::kDefaultDimBits,
                hyperstream::config::kDefaultCapacity);
    // Default execution path: single representative benchmark scenario (~0.3s)
    std::printf("Cluster/default,dim_bits=%d,capacity=%d,updates=%d\n", 10000, 16, 100);
    std::fflush(stdout);
    run_one_dim<10000>();
  } else {
    // Future: parse arguments for custom scenarios; for now, run a broader set
    run_one_dim<10000>();
    run_one_dim<16384>();
    run_one_dim<65536>();
  }

  return EXIT_SUCCESS;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fprintf(stderr, "ERROR: unknown\n");
  return EXIT_FAILURE;
}
