// HyperStream Cluster Memory microbenchmark
// Measures ClusterMemory<Dim,Capacity>::Update() and Finalize() throughput.
// Reports (CSV default):
// name,dim_bits,capacity,updates,update_iters,update_secs,updates_per_sec,finalize_iters,finalize_secs,finalizes_per_sec
// NDJSON mode (--json): one line per sample with fields incl. sample_index,warmup_ms,measure_ms

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <numeric>
#include <vector>

#include "hyperstream/backend/capability.hpp"
#include "hyperstream/backend/policy.hpp"
#include "hyperstream/config.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"

using hyperstream::core::HyperVector;
using hyperstream::memory::ClusterMemory;

namespace {

struct Settings {
  int warmup_ms = 0;  // preserve legacy behavior when omitted
  int measure_ms = 150;
  int samples = 1;
  bool json = false;
};

static Settings ParseArgs(int argc, char** argv) {
  Settings s;
  for (int i = 1; i < argc; ++i) {
    const char* a = argv[i];
    if (std::strncmp(a, "--warmup_ms=", 12) == 0) {
      s.warmup_ms = std::atoi(a + 12);
    } else if (std::strncmp(a, "--measure_ms=", 13) == 0) {
      s.measure_ms = std::atoi(a + 13);
    } else if (std::strncmp(a, "--samples=", 10) == 0) {
      s.samples = std::max(1, std::atoi(a + 10));
    } else if (std::strncmp(a, "--json", 6) == 0) {
      s.json = true;
    }
  }
  return s;
}

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
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() <
           min_ms);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  return {iters, secs};
}

static void print_json_sample(const char* name, std::size_t dim_bits, std::size_t capacity,
                              std::size_t updates, std::size_t iters_u, double secs_u,
                              double updates_ps, std::size_t iters_f, double secs_f,
                              double finalizes_ps, int sample_index, const Settings& s) {
  std::printf(
      "{\"name\":\"%s\",\"dim_bits\":%zu,\"capacity\":%zu,\"updates\":%zu,\"update_iters\":%zu,"
      "\"update_secs\":%.6f,\"updates_per_sec\":%.1f,\"finalize_iters\":%zu,\"finalize_secs\":%.6f,"
      "\"finalizes_per_sec\":%.1f,\"sample_index\":%d,\"warmup_ms\":%d,\"measure_ms\":%d}\n",
      name, dim_bits, capacity, updates, iters_u, secs_u, updates_ps, iters_f, secs_f, finalizes_ps,
      sample_index, s.warmup_ms, s.measure_ms);
}

template <std::size_t Dim, std::size_t Capacity>
static void bench_cluster(const char* name, std::size_t updates, const Settings& s) {
  ClusterMemory<Dim, Capacity> cmem;
  std::uint64_t seed = 1;

  auto do_update_loop = [&](volatile std::uint64_t* sink) {
    for (std::size_t i = 0; i < updates; ++i) {
      HyperVector<Dim, bool> hv;
      hv.Clear();
      fill_random(&hv, seed);
      (void)cmem.Update(42, hv);
      *sink ^= hv.Words()[0];
    }
  };
  HyperVector<Dim, bool> out;
  auto do_finalize_loop = [&](volatile std::uint64_t* sink) {
    cmem.Finalize(42, &out);
    *sink ^= out.Words()[0];
  };

  // Warmup (optional)
  if (s.warmup_ms > 0) {
    (void)run_for_ms(do_update_loop, s.warmup_ms);
    (void)run_for_ms(do_finalize_loop, s.warmup_ms);
  }

  std::vector<double> upd_ps_v;
  upd_ps_v.reserve(static_cast<std::size_t>(s.samples));
  std::vector<double> fin_ps_v;
  fin_ps_v.reserve(static_cast<std::size_t>(s.samples));

  for (int si = 0; si < s.samples; ++si) {
    auto [iters_u, secs_u] = run_for_ms(do_update_loop, s.measure_ms);
    auto [iters_f, secs_f] = run_for_ms(do_finalize_loop, s.measure_ms);

    const double updates_ps = static_cast<double>(iters_u) * updates / secs_u;
    const double finalizes_ps = static_cast<double>(iters_f) / secs_f;
    upd_ps_v.push_back(updates_ps);
    fin_ps_v.push_back(finalizes_ps);

    if (s.json) {
      print_json_sample(name, Dim, Capacity, updates, iters_u, secs_u, updates_ps, iters_f, secs_f,
                        finalizes_ps, si, s);
    } else {
      // Emit per-phase early line only for first sample to preserve legacy behavior
      if (si == 0) {
        std::printf(
            "%s-update,dim_bits=%zu,capacity=%zu,updates=%zu,update_iters=%zu,update_secs=%.6f,"
            "updates_per_sec=%.1f\n",
            name, Dim, Capacity, updates, iters_u, secs_u, updates_ps);
        std::fflush(stdout);
      }
      // Combined line
      if (s.samples == 1) {
        std::printf(
            "%s,dim_bits=%zu,capacity=%zu,updates=%zu,update_iters=%zu,update_secs=%.6f,updates_"
            "per_sec=%.1f,finalize_iters=%zu,finalize_secs=%.6f,finalizes_per_sec=%.1f\n",
            name, Dim, Capacity, updates, iters_u, secs_u, updates_ps, iters_f, secs_f,
            finalizes_ps);
      } else {
        std::printf(
            "%s,dim_bits=%zu,capacity=%zu,updates=%zu,sample=%d,update_iters=%zu,update_secs=%.6f,"
            "updates_per_sec=%.1f,finalize_iters=%zu,finalize_secs=%.6f,finalizes_per_sec=%.1f\n",
            name, Dim, Capacity, updates, si, iters_u, secs_u, updates_ps, iters_f, secs_f,
            finalizes_ps);
      }
      std::fflush(stdout);
    }
  }

  if (s.samples > 1) {
    auto agg = [](std::vector<double> v) {
      std::sort(v.begin(), v.end());
      const double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
      const double median =
          (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
      double ss = 0.0;
      for (double x : v) {
        double d = x - mean;
        ss += d * d;
      }
      const double stdev = std::sqrt(ss / v.size());
      return std::tuple<double, double, double>(mean, median, stdev);
    };
    auto [u_mean, u_med, u_std] = agg(upd_ps_v);
    auto [f_mean, f_med, f_std] = agg(fin_ps_v);
    if (s.json) {
      std::printf(
          "{\"name\":\"%s\",\"dim_bits\":%zu,\"capacity\":%zu,\"updates\":%zu,\"aggregate\":true,"
          "\"samples\":%d,\"updates_per_sec\":{\"mean\":%.1f,\"median\":%.1f,\"stdev\":%.1f},"
          "\"finalizes_per_sec\":{\"mean\":%.1f,\"median\":%.1f,\"stdev\":%.1f},\"warmup_ms\":%d,"
          "\"measure_ms\":%d}\n",
          name, Dim, Capacity, updates, s.samples, u_mean, u_med, u_std, f_mean, f_med, f_std,
          s.warmup_ms, s.measure_ms);
    } else {
      std::printf(
          "%s,dim_bits=%zu,capacity=%zu,updates=%zu,aggregate=samples:%d,updates_ps_mean=%.1f,"
          "updates_ps_median=%.1f,updates_ps_stdev=%.1f,finalizes_ps_mean=%.1f,finalizes_ps_median="
          "%.1f,finalizes_ps_stdev=%.1f\n",
          name, Dim, Capacity, updates, s.samples, u_mean, u_med, u_std, f_mean, f_med, f_std);
    }
  }
}

template <std::size_t Dim>
static void run_one_dim(const Settings& s) {
  bench_cluster<Dim, 16>("Cluster/update_finalize", 100, s);
}

}  // namespace

int main(int argc, char** argv) try {
  // Make stdout/stderr unbuffered to ensure visibility across environments
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);

  const Settings s = ParseArgs(argc, argv);

  if (argc == 1) {
    // Print active configuration
    std::printf("Config/profile=%s,default_dim_bits=%zu,default_capacity=%zu\n",
                hyperstream::config::kActiveProfile, hyperstream::config::kDefaultDimBits,
                hyperstream::config::kDefaultCapacity);
    // Default execution path: single representative benchmark scenario (~0.3s)
    std::printf("Cluster/default,dim_bits=%d,capacity=%d,updates=%d\n", 10000, 16, 100);
    std::fflush(stdout);
    run_one_dim<10000>(s);
  } else {
    // Custom/extended execution path
    run_one_dim<10000>(s);
    run_one_dim<16384>(s);
    run_one_dim<65536>(s);
  }

  return EXIT_SUCCESS;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fprintf(stderr, "ERROR: unknown\n");
  return EXIT_FAILURE;
}
