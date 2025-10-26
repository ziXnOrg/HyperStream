#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/item_memory.hpp"
#include "hyperstream/encoding/numeric.hpp"
#include "hyperstream/encoding/symbol.hpp"
#include "hyperstream/memory/associative.hpp"

namespace {

using hyperstream::core::Bind;
using hyperstream::core::HyperVector;
using hyperstream::encoding::ItemMemory;
using hyperstream::encoding::RandomProjectionEncoder;
using hyperstream::encoding::SymbolEncoder;
using hyperstream::encoding::ThermometerEncoder;
using hyperstream::memory::ClusterMemory;
using hyperstream::memory::PrototypeMemory;

static std::string TestsDir() {
#ifdef HYPERSTREAM_TESTS_DIR
  return std::string(HYPERSTREAM_TESTS_DIR);
#else
  return std::string("tests");
#endif
}

static const char* BackendId() {
#ifdef HYPERSTREAM_FORCE_SCALAR
  return "scalar";
#else
  return "simd";
#endif
}

// 64-bit FNV-1a over words
static inline std::uint64_t HashWords(const std::uint64_t* words, std::size_t n) {
  const std::uint64_t kOffset = 1469598103934665603ull;
  const std::uint64_t kPrime = 1099511628211ull;
  std::uint64_t h = kOffset;
  for (std::size_t i = 0; i < n; ++i) {
    std::uint64_t w = words[i];
    for (int b = 0; b < 8; ++b) {
      std::uint8_t byte = static_cast<std::uint8_t>(w & 0xFF);
      h ^= byte;
      h *= kPrime;
      w >>= 8;
    }
  }
  return h;
}

static std::string Hex64(std::uint64_t x) {
  char buf[19];
  std::snprintf(buf, sizeof(buf), "0x%016llx", static_cast<unsigned long long>(x));
  return std::string(buf);
}

struct Event {
  int v = 1;
  std::uint64_t seq = 0;
  std::string src;
  std::string eid;
  std::string kind;        // symbol|numeric|vector|label
  std::int64_t ts_ms = 0;  // informational only
  // payload (only one active by kind)
  std::string sym;
  double val = 0.0;
  std::vector<float> vec;
  std::string label;
};

// Tiny ad-hoc NDJSON parser for the canonical schema (no external deps)
static bool ParseEventLine(const std::string& line, Event* ev) {
  auto find_str = [&](const char* key, std::string* out) -> bool {
    const std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    if (line[p] == '"')
      ++p;
    else
      return false;
    auto q = line.find('"', p);
    if (q == std::string::npos) return false;
    *out = line.substr(p, q - p);
    return true;
  };
  auto find_num_i64 = [&](const char* key, std::int64_t* out) -> bool {
    const std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    // read until comma or }
    auto q = p;
    while (q < line.size() && (line[q] == '-' || (line[q] >= '0' && line[q] <= '9'))) ++q;
    *out = std::stoll(line.substr(p, q - p));
    return true;
  };
  auto find_num_double = [&](const char* key, double* out) -> bool {
    const std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    auto q = p;
    while (q < line.size() &&
           (line[q] == '-' || line[q] == '.' || (line[q] >= '0' && line[q] <= '9')))
      ++q;
    *out = std::stod(line.substr(p, q - p));
    return true;
  };

  // Required fields
  if (!find_num_i64("v", reinterpret_cast<std::int64_t*>(&ev->v))) return false;
  if (!find_num_i64("seq", reinterpret_cast<std::int64_t*>(&ev->seq))) return false;
  if (!find_str("src", &ev->src)) return false;
  if (!find_str("eid", &ev->eid)) return false;
  if (!find_str("kind", &ev->kind)) return false;
  (void)find_num_i64("ts_ms", &ev->ts_ms);

  // Payload by kind
  if (ev->kind == "symbol") {
    (void)find_str("sym", &ev->sym);
  } else if (ev->kind == "numeric") {
    (void)find_num_double("val", &ev->val);
  } else if (ev->kind == "vector") {
    // parse "vec":[...]
    const std::string k = "\"vec\":[";
    auto p = line.find(k);
    if (p != std::string::npos) {
      p += k.size();
      ev->vec.clear();
      while (p < line.size() && line[p] != ']') {
        // skip spaces
        while (p < line.size() && (line[p] == ' ' || line[p] == ',')) ++p;
        auto q = p;
        while (q < line.size() &&
               (line[q] == '-' || line[q] == '.' || (line[q] >= '0' && line[q] <= '9')))
          ++q;
        if (q > p) {
          ev->vec.push_back(static_cast<float>(std::stod(line.substr(p, q - p))));
          p = q;
        } else {
          break;
        }
      }
    }
  } else if (ev->kind == "label") {
    (void)find_str("label", &ev->label);
  }
  return true;
}

// Ordered tuple comparator for (seq, src, eid)
static bool TotalOrderLt(const Event& a, const Event& b) {
  if (a.seq != b.seq) return a.seq < b.seq;
  if (a.src != b.src) return a.src < b.src;
  return a.eid < b.eid;
}

// Compute checkpoint hashes at every K events and final, returning vector of hashes
struct StreamResult {
  std::vector<std::uint64_t> checkpoints;  // at K, 2K, ...
  std::uint64_t final_hash = 0;
};

// Deterministic streaming pipeline over events with fixed encoders and memories
// Chunker yields next slice size given remaining count (strategies implemented in tests)
static StreamResult IngestStream(const std::vector<Event>& ordered,
                                 std::function<std::size_t(std::size_t)> next_chunk,
                                 int K /*checkpoint interval*/) {
  static constexpr std::size_t D = 256;
  // Fixed encoder parameters
  SymbolEncoder<D> sym(0x9e3779b97f4a7c15ull);
  ThermometerEncoder<D> therm(0.0, 100.0);
  RandomProjectionEncoder<D> proj(0x51ed2701f3a5c7b9ull);
  ItemMemory<D> item(0x123456789abcdef0ull);

  PrototypeMemory<D, 16> pmem;
  ClusterMemory<D, 4> cmem;  // use label 1 as observation cluster

  HyperVector<D, bool> last_obs;
  last_obs.Clear();
  HyperVector<D, bool> hv, out;

  std::uint64_t mix = 0;  // rolling mix to include prototype words too
  StreamResult res;

  std::size_t i = 0;
  while (i < ordered.size()) {
    const std::size_t remain = ordered.size() - i;
    std::size_t take = next_chunk(remain);
    if (take == 0) take = 1;
    if (take > remain) take = remain;

    for (std::size_t j = 0; j < take; ++j) {
      const Event& ev = ordered[i + j];
      if (ev.kind == "symbol") {
        sym.EncodeToken(ev.sym, &hv);
        (void)cmem.Update(1, hv);
        last_obs = hv;
      } else if (ev.kind == "numeric") {
        therm.Encode(ev.val, &hv);
        (void)cmem.Update(1, hv);
        last_obs = hv;
      } else if (ev.kind == "vector") {
        // normalize vector length by truncating/padding zeros
        const std::size_t n = ev.vec.size();
        const std::size_t use = n;  // RandomProjection takes arbitrary length
        if (use > 0) {
          proj.Encode(ev.vec.data(), static_cast<int>(use), &hv);
        } else {
          hv.Clear();
        }
        (void)cmem.Update(1, hv);
        last_obs = hv;
      } else if (ev.kind == "label") {
        // Bind last observation to label HV and learn into prototypes
        HyperVector<D, bool> hv_label;
        item.EncodeToken(ev.label, &hv_label);
        HyperVector<D, bool> bound;
        Bind(last_obs, hv_label, &bound);
        const std::uint64_t label_id =
            hyperstream::encoding::detail_itemmemory::Fnv1a64(ev.label, 0xfeedf00dULL);
        (void)pmem.Learn(label_id, bound);
      }

      // Mix some prototype words into rolling state for stronger coverage
      if (pmem.Size() > 0) {
        const auto* entries = pmem.Data();
        const auto& words = entries[pmem.Size() - 1].hv.Words();
        mix ^= words[0];
      }

      // Checkpoint
      const std::size_t idx = i + j + 1;
      if (K > 0 && (idx % K) == 0) {
        cmem.Finalize(1, &out);
        const auto h = HashWords(out.Words().data(), out.Words().size()) ^ mix;
        res.checkpoints.push_back(h);
      }
    }

    i += take;
  }

  cmem.Finalize(1, &out);
  res.final_hash = HashWords(out.Words().data(), out.Words().size()) ^ mix;
  return res;
}

static std::vector<Event> LoadCanonicalEvents() {
  std::vector<Event> evs;
  const std::string path = TestsDir() + "/golden/streaming_events.ndjson";
  std::ifstream f(path);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    Event ev;
    if (ParseEventLine(line, &ev)) evs.push_back(std::move(ev));
  }
  return evs;
}

static std::string ReadTextFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::ostringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

static bool ExtractBackendHashes(const std::string& json, const char* backend,
                                 std::vector<std::string>* checkpoints, std::string* final_hex) {
  const std::string dkey = "\"D256\"";
  const std::string bkey = std::string("\"") + backend + "\"";
  auto dpos = json.find(dkey);
  if (dpos == std::string::npos) return false;
  auto bpos = json.find(bkey, dpos);
  if (bpos == std::string::npos) return false;
  auto cpos = json.find("\"checkpoints\"", bpos);
  if (cpos == std::string::npos) return false;
  auto apos = json.find('[', cpos);
  auto aend = json.find(']', apos);
  if (apos == std::string::npos || aend == std::string::npos) return false;
  checkpoints->clear();
  std::size_t p = apos + 1;
  while (true) {
    auto q1 = json.find('"', p);
    if (q1 == std::string::npos || q1 >= aend) break;
    auto q2 = json.find('"', q1 + 1);
    if (q2 == std::string::npos || q2 > aend) break;
    checkpoints->push_back(json.substr(q1 + 1, q2 - q1 - 1));
    p = q2 + 1;
  }
  auto fpos = json.find("\"final\"", aend);
  if (fpos == std::string::npos) return false;
  auto q1 = json.find('"', fpos + 7);
  auto q2 = json.find('"', q1 + 1);
  if (q1 == std::string::npos || q2 == std::string::npos) return false;
  *final_hex = json.substr(q1 + 1, q2 - q1 - 1);
  return true;
}

// --- Tests ---

TEST(StreamingDeterminism, ChunkingInvariance_1_8_64_Random) {
  ::testing::Test::RecordProperty("backend", BackendId());
  auto base = LoadCanonicalEvents();
  // Ensure canonical is already in total order
  std::vector<Event> ord = base;
  std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  ASSERT_EQ(ord.size(), base.size());
  for (std::size_t i = 0; i < ord.size(); ++i) {
    EXPECT_EQ(ord[i].seq, base[i].seq);
  }

  auto strat1 = [](std::size_t) { return std::size_t(1); };
  auto strat8 = [](std::size_t) { return std::size_t(8); };
  auto strat64 = [](std::size_t) { return std::size_t(64); };
  std::mt19937 rng(12345);
  auto stratR = [&](std::size_t rem) {
    std::uniform_int_distribution<int> d(1, 64);
    return (std::size_t)(1 + ((d(rng) + static_cast<int>(rem % 4)) % 64));
  };

  const int K = 16;
  const auto r1 = IngestStream(ord, strat1, K);
  const auto r8 = IngestStream(ord, strat8, K);
  const auto r64 = IngestStream(ord, strat64, K);
  const auto rR = IngestStream(ord, stratR, K);

  ASSERT_EQ(r1.checkpoints.size(), r8.checkpoints.size());
  ASSERT_EQ(r1.checkpoints.size(), r64.checkpoints.size());
  ASSERT_EQ(r1.checkpoints.size(), rR.checkpoints.size());
  for (std::size_t i = 0; i < r1.checkpoints.size(); ++i) {
    EXPECT_EQ(r1.checkpoints[i], r8.checkpoints[i]);
    EXPECT_EQ(r1.checkpoints[i], r64.checkpoints[i]);
    EXPECT_EQ(r1.checkpoints[i], rR.checkpoints[i]);
  }
  EXPECT_EQ(r1.final_hash, r8.final_hash);
  EXPECT_EQ(r1.final_hash, r64.final_hash);
  EXPECT_EQ(r1.final_hash, rR.final_hash);
}

TEST(StreamingDeterminism, InterleaveParity_PremergedVsSortedMerge) {
  auto base = LoadCanonicalEvents();
  std::vector<Event> premerged = base;  // canonical is premerged
  // Simulate re-merge by sorting a copy
  std::vector<Event> merged = base;
  std::stable_sort(merged.begin(), merged.end(), TotalOrderLt);
  auto strat8 = [](std::size_t) { return std::size_t(8); };
  const int K = 16;
  const auto r0 = IngestStream(premerged, strat8, K);
  const auto r1 = IngestStream(merged, strat8, K);
  ASSERT_EQ(r0.checkpoints.size(), r1.checkpoints.size());
  for (std::size_t i = 0; i < r0.checkpoints.size(); ++i)
    EXPECT_EQ(r0.checkpoints[i], r1.checkpoints[i]);
  EXPECT_EQ(r0.final_hash, r1.final_hash);
}

TEST(StreamingDeterminism, GoldenParity_CheckpointsAndFinal) {
  ::testing::Test::RecordProperty("backend", BackendId());
  auto base = LoadCanonicalEvents();
  std::vector<Event> ord = base;
  std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  const int K = 16;
  auto strat8 = [](std::size_t) { return std::size_t(8); };
  const auto r = IngestStream(ord, strat8, K);
  const std::string json = ReadTextFile(TestsDir() + "/golden/streaming_hashes.json");
  std::vector<std::string> exp_chk;
  std::string exp_final;
  ASSERT_TRUE(ExtractBackendHashes(json, BackendId(), &exp_chk, &exp_final));
  ASSERT_EQ(exp_chk.size(), r.checkpoints.size());
  for (std::size_t i = 0; i < r.checkpoints.size(); ++i) {
    EXPECT_EQ(Hex64(r.checkpoints[i]), exp_chk[i]) << "chkpt index=" << i;
  }
  EXPECT_EQ(Hex64(r.final_hash), exp_final);
}

TEST(StreamingDeterminism, OutOfOrderArrival_WithReorderBuffer) {
  auto base = LoadCanonicalEvents();
  // introduce bounded jitter then sort by total order
  std::vector<Event> jitter = base;
  if (jitter.size() > 8) {
    for (std::size_t i = 0; i + 4 < jitter.size(); i += 5) {
      std::swap(jitter[i], jitter[i + 4]);
    }
  }
  std::stable_sort(jitter.begin(), jitter.end(), TotalOrderLt);
  auto stratR = [](std::size_t rem) {
    (void)rem;
    return std::size_t(1 + (rem % 7));
  };
  const int K = 16;
  const auto r0 = IngestStream(base, stratR, K);
  const auto r1 = IngestStream(jitter, stratR, K);
  ASSERT_EQ(r0.checkpoints.size(), r1.checkpoints.size());
  for (std::size_t i = 0; i < r0.checkpoints.size(); ++i)
    EXPECT_EQ(r0.checkpoints[i], r1.checkpoints[i]);
  EXPECT_EQ(r0.final_hash, r1.final_hash);
}

// Epoch/window boundaries == every K events; verify checkpoint equality across chunkings
TEST(StreamingDeterminism, EpochWindow_CheckpointsEqual) {
  auto base = LoadCanonicalEvents();
  std::vector<Event> ord = base;
  std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  const int K = 16;
  auto strat1 = [](std::size_t) { return std::size_t(1); };
  auto strat64 = [](std::size_t) { return std::size_t(64); };
  const auto r1 = IngestStream(ord, strat1, K);
  const auto r2 = IngestStream(ord, strat64, K);
  ASSERT_EQ(r1.checkpoints.size(), r2.checkpoints.size());
  for (std::size_t i = 0; i < r1.checkpoints.size(); ++i)
    EXPECT_EQ(r1.checkpoints[i], r2.checkpoints[i]);
}

// Disabled generator to dump canonical streaming hashes for golden file
TEST(StreamingDeterminism, DISABLED_DumpStreamingHashes) {
  ::testing::Test::RecordProperty("backend", BackendId());
  auto base = LoadCanonicalEvents();
  std::vector<Event> ord = base;
  std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  const int K = 16;
  auto strat = [](std::size_t) { return std::size_t(8); };
  const auto r = IngestStream(ord, strat, K);
  std::printf("{\"suite\":\"Streaming\",\"dim\":256,\"backend\":\"%s\",\"chkpt\":[", BackendId());
  for (std::size_t i = 0; i < r.checkpoints.size(); ++i) {
    std::printf("%s\"%s\"", (i == 0 ? "" : ","), Hex64(r.checkpoints[i]).c_str());
  }
  std::printf("],\"final\":\"%s\"}\n", Hex64(r.final_hash).c_str());
}

}  // namespace
