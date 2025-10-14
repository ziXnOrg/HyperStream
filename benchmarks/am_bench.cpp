// HyperStream Associative Memory microbenchmark
// Measures PrototypeMemory<Dim,Capacity>::Classify() throughput vs number of entries.
// Reports (CSV default): name,dim_bits,capacity,size,iters,secs,queries_per_sec,eff_gb_per_sec
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
#include <utility>
#include <vector>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

#include "hyperstream/backend/policy.hpp"
#include "hyperstream/config.hpp"
#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"
#if HS_X86_ARCH
#include "hyperstream/backend/cpu_backend_avx2.hpp"
#include "hyperstream/backend/cpu_backend_sse2.hpp"
#endif
#if HS_ARM64_ARCH
#include "hyperstream/backend/cpu_backend_neon.hpp"
#endif
#include "hyperstream/backend/capability.hpp"

using hyperstream::core::HyperVector;
using hyperstream::memory::PrototypeMemory;

namespace {

struct Settings {
  int warmup_ms = 0;  // preserve legacy behavior when omitted
  int measure_ms = 300;
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
  } while (std::chrono::duration_cast<std::chrono::milliseconds>(clock::now() - t0).count() <
           min_ms);
  const double secs = std::chrono::duration<double>(clock::now() - t0).count();
  return {iters, secs};
}

static void print_json_sample(const char* name, std::size_t dim_bits, std::size_t capacity,
                              std::size_t size, std::size_t iters, double secs, double qps,
                              double gbps, int sample_index, const Settings& s) {
  std::printf(
      "{\"name\":\"%s\",\"dim_bits\":%zu,\"capacity\":%zu,\"size\":%zu,\"iters\":%zu,\"secs\":%.6f,"
      "\"queries_per_sec\":%.1f,\"eff_gb_per_sec\":%.3f,\"sample_index\":%d,\"warmup_ms\":%d,"
      "\"measure_ms\":%d}\n",
      name, dim_bits, capacity, size, iters, secs, qps, gbps, sample_index, s.warmup_ms,
      s.measure_ms);
}

template <std::size_t Dim, std::size_t Capacity, typename ClassifyFn>
static void bench_am(const char* name, std::size_t size, ClassifyFn&& classify, const Settings& s) {
  PrototypeMemory<Dim, Capacity> am;
  // Fill entries deterministically up to 'size'
  std::uint64_t seed = 12345;
  for (std::size_t i = 0; i < size && i < Capacity; ++i) {
    HyperVector<Dim, bool> hv;
    hv.Clear();
    fill_random(&hv, seed);
    (void)am.Learn(i + 1, hv);
  }
  HyperVector<Dim, bool> query;
  query.Clear();
  fill_random(&query, seed);

  const std::size_t words = HyperVector<Dim, bool>::WordCount();
  const std::size_t bytes_per_iter = (size * words + words) * sizeof(std::uint64_t);  // approx

  // Warmup (optional)
  if (s.warmup_ms > 0) {
    (void)run_for_ms(
        [&](volatile std::uint64_t* sink) {
          const auto lbl = classify(am, query);
          *sink ^= lbl;
        },
        s.warmup_ms);
  }

  // Samples
  std::vector<double> qps_v;
  qps_v.reserve(static_cast<std::size_t>(s.samples));
  std::vector<double> gbps_v;
  gbps_v.reserve(static_cast<std::size_t>(s.samples));
  for (int si = 0; si < s.samples; ++si) {
    auto [iters, secs] = run_for_ms(
        [&](volatile std::uint64_t* sink) {
          const auto lbl = classify(am, query);
          *sink ^= lbl;
        },
        s.measure_ms);
    const double qps = static_cast<double>(iters) / secs;
    const double eff_gbps = (static_cast<double>(bytes_per_iter) * iters / secs) / 1e9;
    qps_v.push_back(qps);
    gbps_v.push_back(eff_gbps);
    if (s.json) {
      print_json_sample(name, Dim, Capacity, size, iters, secs, qps, eff_gbps, si, s);
    } else {
      // Preserve legacy line when samples==1 and no JSON
      if (s.samples == 1) {
        std::printf(
            "%s,dim_bits=%zu,capacity=%zu,size=%zu,iters=%zu,secs=%.6f,queries_per_sec=%.1f,eff_gb_"
            "per_sec=%.3f\n",
            name, Dim, Capacity, size, iters, secs, qps, eff_gbps);
      } else {
        std::printf(
            "%s,dim_bits=%zu,capacity=%zu,size=%zu,sample=%d,iters=%zu,secs=%.6f,queries_per_sec=%."
            "1f,eff_gb_per_sec=%.3f\n",
            name, Dim, Capacity, size, si, iters, secs, qps, eff_gbps);
      }
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
    auto [q_mean, q_med, q_std] = agg(qps_v);
    auto [g_mean, g_med, g_std] = agg(gbps_v);
    if (s.json) {
      std::printf(
          "{\"name\":\"%s\",\"dim_bits\":%zu,\"capacity\":%zu,\"size\":%zu,\"aggregate\":true,"
          "\"samples\":%d,\"queries_per_sec\":{\"mean\":%.1f,\"median\":%.1f,\"stdev\":%.1f},\"eff_"
          "gb_per_sec\":{\"mean\":%.3f,\"median\":%.3f,\"stdev\":%.3f},\"warmup_ms\":%d,\"measure_"
          "ms\":%d}\n",
          name, Dim, Capacity, size, s.samples, q_mean, q_med, q_std, g_mean, g_med, g_std,
          s.warmup_ms, s.measure_ms);
    } else {
      std::printf(
          "%s,dim_bits=%zu,capacity=%zu,size=%zu,aggregate=samples:%d,qps_mean=%.1f,qps_median=%."
          "1f,qps_stdev=%.1f,gbps_mean=%.3f,gbps_median=%.3f,gbps_stdev=%.3f\n",
          name, Dim, Capacity, size, s.samples, q_mean, q_med, q_std, g_mean, g_med, g_std);
    }
  }
}

template <std::size_t Dim>
static void run_one_dim(const Settings& s) {
  // Choose representative capacities/sizes
  bench_am<Dim, 256>("AM/core", 256, [](auto& am, const auto& q) { return am.Classify(q, 0); }, s);
  bench_am<Dim, 1024>(
      "AM/core", 1024, [](auto& am, const auto& q) { return am.Classify(q, 0); }, s);

  // SSE2 distance functor
#if HS_X86_ARCH
  bench_am<Dim, 1024>(
      "AM/sse2", 1024,
      [](const auto& am, const auto& q) {
        return am.Classify(
            q,
            [](const auto& a, const auto& b) {
              return hyperstream::backend::sse2::HammingDistanceSSE2<Dim>(a, b);
            },
            0);
      },
      s);
#endif

  // AVX2 distance functor
#if HS_ARM64_ARCH
  bench_am<Dim, 1024>(
      "AM/neon", 1024,
      [](const auto& am, const auto& q) {
        return am.Classify(
            q,
            [](const auto& a, const auto& b) {
              return hyperstream::backend::neon::HammingDistanceNEON<Dim>(a, b);
            },
            0);
      },
      s);
#endif

#if defined(__AVX2__)
  bench_am<Dim, 1024>(
      "AM/avx2", 1024,
      [](const auto& am, const auto& q) {
        return am.Classify(
            q,
            [](const auto& a, const auto& b) {
              return hyperstream::backend::avx2::HammingDistanceAVX2<Dim>(a, b);
            },
            0);
      },
      s);
#endif
}

}  // namespace

int main(int argc, char** argv) try {
  // Make stdout/stderr unbuffered to ensure visibility across environments
  setvbuf(stdout, nullptr, _IONBF, 0);
  setvbuf(stderr, nullptr, _IONBF, 0);

  const Settings s = ParseArgs(argc, argv);

  // Print active configuration
  std::printf("Config/profile=%s,default_dim_bits=%zu,default_capacity=%zu\n",
              hyperstream::config::kActiveProfile, hyperstream::config::kDefaultDimBits,
              hyperstream::config::kDefaultCapacity);

  // Use a typical binary dimension for HDC
  run_one_dim<10000>(s);
  run_one_dim<16384>(s);
  run_one_dim<65536>(s);

  return EXIT_SUCCESS;
} catch (const std::exception& e) {
  std::fprintf(stderr, "ERROR: %s\n", e.what());
  return EXIT_FAILURE;
} catch (...) {
  std::fprintf(stderr, "ERROR: unknown\n");
  return EXIT_FAILURE;
}
