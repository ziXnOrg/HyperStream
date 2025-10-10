#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/item_memory.hpp"
#include "hyperstream/encoding/symbol.hpp"
#include "hyperstream/encoding/numeric.hpp"
#include "hyperstream/memory/associative.hpp"
#include "hyperstream/io/serialization.hpp"

namespace {

using hyperstream::core::Bind;
using hyperstream::core::HyperVector;
using hyperstream::encoding::ItemMemory;
using hyperstream::encoding::SymbolEncoder;
using hyperstream::encoding::ThermometerEncoder;
using hyperstream::encoding::RandomProjectionEncoder;
using hyperstream::memory::PrototypeMemory;
using hyperstream::memory::ClusterMemory;
using hyperstream::io::SavePrototype;
using hyperstream::io::LoadPrototype;
using hyperstream::io::SaveCluster;
using hyperstream::io::LoadCluster;

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
  std::string kind; // symbol|numeric|vector|label
  std::int64_t ts_ms = 0; // informational only
  // payload
  std::string sym;
  double val = 0.0;
  std::vector<float> vec;
  std::string label;
};

// Minimal ad-hoc NDJSON parser for canonical schema
static bool ParseEventLine(const std::string& line, Event* ev) {
  auto find_str = [&](const char* key, std::string* out)->bool{
    const std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    if (line[p] == '"') ++p; else return false;
    auto q = line.find('"', p);
    if (q == std::string::npos) return false;
    *out = line.substr(p, q - p);
    return true;
  };
  auto find_num_i64 = [&](const char* key, std::int64_t* out)->bool{
    const std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    auto q = p;
    while (q < line.size() && (line[q] == '-' || (line[q] >= '0' && line[q] <= '9'))) ++q;
    *out = std::stoll(line.substr(p, q - p));
    return true;
  };
  auto find_num_double = [&](const char* key, double* out)->bool{
    const std::string k = std::string("\"") + key + "\":";
    auto p = line.find(k);
    if (p == std::string::npos) return false;
    p += k.size();
    auto q = p;
    while (q < line.size() && (line[q] == '-' || line[q] == '.' || (line[q] >= '0' && line[q] <= '9'))) ++q;
    *out = std::stod(line.substr(p, q - p));
    return true;
  };

  if (!find_num_i64("v", reinterpret_cast<std::int64_t*>(&ev->v))) return false;
  if (!find_num_i64("seq", reinterpret_cast<std::int64_t*>(&ev->seq))) return false;
  if (!find_str("src", &ev->src)) return false;
  if (!find_str("eid", &ev->eid)) return false;
  if (!find_str("kind", &ev->kind)) return false;
  (void)find_num_i64("ts_ms", &ev->ts_ms);

  if (ev->kind == "symbol") {
    (void)find_str("sym", &ev->sym);
  } else if (ev->kind == "numeric") {
    (void)find_num_double("val", &ev->val);
  } else if (ev->kind == "vector") {
    const std::string k = "\"vec\":";
    auto p = line.find(k);
    if (p != std::string::npos) {
      p = line.find('[', p);
      if (p != std::string::npos) {
        ++p;
        ev->vec.clear();
        while (p < line.size() && line[p] != ']') {
          while (p < line.size() && (line[p] == ' ' || line[p] == ',')) ++p;
          auto q = p;
          while (q < line.size() && (line[q] == '-' || line[q] == '.' || (line[q] >= '0' && line[q] <= '9'))) ++q;
          if (q > p) { ev->vec.push_back(static_cast<float>(std::stod(line.substr(p, q - p)))); p = q; } else { break; }
        }
      }
    }
  } else if (ev->kind == "label") {
    (void)find_str("label", &ev->label);
  }
  return true;
}

static bool TotalOrderLt(const Event& a, const Event& b) {
  if (a.seq != b.seq) return a.seq < b.seq;
  if (a.src != b.src) return a.src < b.src;
  return a.eid < b.eid;
}

struct StreamResult { std::vector<std::uint64_t> checkpoints; std::uint64_t final_hash = 0; };

// Stateful pipeline for snapshot/restore
struct Pipeline {
  static constexpr std::size_t D = 256;
  SymbolEncoder<D> sym{0x9e3779b97f4a7c15ull};
  ThermometerEncoder<D> therm{0.0, 100.0};
  RandomProjectionEncoder<D> proj{0x51ed2701f3a5c7b9ull};
  ItemMemory<D> item{0x123456789abcdef0ull};
  PrototypeMemory<D, 16> pmem;
  ClusterMemory<D, 4> cmem;
  HyperVector<D, bool> last_obs;
  std::uint64_t mix = 0;
  int K = 16;

  Pipeline() { last_obs.Clear(); }

  void Process(const Event& ev) {
    HyperVector<D, bool> hv;
    if (ev.kind == "symbol") {
      sym.EncodeToken(ev.sym, &hv);
      (void)cmem.Update(1, hv);
      last_obs = hv;
    } else if (ev.kind == "numeric") {
      therm.Encode(ev.val, &hv);
      (void)cmem.Update(1, hv);
      last_obs = hv;
    } else if (ev.kind == "vector") {
      if (!ev.vec.empty()) {
        proj.Encode(ev.vec.data(), static_cast<int>(ev.vec.size()), &hv);
      } else {
        hv.Clear();
      }
      (void)cmem.Update(1, hv);
      last_obs = hv;
    } else if (ev.kind == "label") {
      HyperVector<D, bool> hv_label; item.EncodeToken(ev.label, &hv_label);
      HyperVector<D, bool> bound; Bind(last_obs, hv_label, &bound);
      const std::uint64_t label_id = hyperstream::encoding::detail_itemmemory::Fnv1a64(ev.label, 0xfeedf00dULL);
      (void)pmem.Learn(label_id, bound);
    }
    if (pmem.size() > 0) {
      const auto* entries = pmem.data();
      const auto& words = entries[pmem.size() - 1].hv.Words();
      mix ^= words[0];
    }
  }

  std::uint64_t CheckpointHash() const {
    HyperVector<D, bool> out; cmem.Finalize(1, &out);
    return HashWords(out.Words().data(), out.Words().size()) ^ mix;
  }
};

static std::vector<Event> LoadCanonicalEvents() {
  std::vector<Event> evs;
  const std::string path = TestsDir() + "/golden/streaming_events.ndjson";
  std::ifstream f(path);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty()) continue;
    Event ev; if (ParseEventLine(line, &ev)) evs.push_back(std::move(ev));
  }
  return evs;
}

static std::string ReadTextFile(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  EXPECT_TRUE(f.good()) << "open failed: " << path;
  std::ostringstream ss; ss << f.rdbuf();
  return ss.str();
}

static bool ExtractBackendHashes(const std::string& json, const char* backend, std::vector<std::string>* checkpoints, std::string* final_hex) {
  const std::string dkey = "\"D256\"";
  const std::string bkey = std::string("\"") + backend + "\"";
  auto dpos = json.find(dkey); if (dpos == std::string::npos) return false;
  auto bpos = json.find(bkey, dpos); if (bpos == std::string::npos) return false;
  auto cpos = json.find("\"checkpoints\"", bpos); if (cpos == std::string::npos) return false;
  auto apos = json.find('[', cpos); auto aend = json.find(']', apos);
  if (apos == std::string::npos || aend == std::string::npos) return false;
  checkpoints->clear(); std::size_t p = apos + 1;
  while (true) { auto q1 = json.find('"', p); if (q1 == std::string::npos || q1 >= aend) break; auto q2 = json.find('"', q1 + 1); if (q2 == std::string::npos || q2 > aend) break; checkpoints->push_back(json.substr(q1 + 1, q2 - q1 - 1)); p = q2 + 1; }
  auto fpos = json.find("\"final\"", aend); if (fpos == std::string::npos) return false; auto q1 = json.find('"', fpos + 7); auto q2 = json.find('"', q1 + 1); if (q1 == std::string::npos || q2 == std::string::npos) return false; *final_hex = json.substr(q1 + 1, q2 - q1 - 1); return true;
}

// Write snapshot fixtures for N   {16,32,48}
TEST(SnapshotRestore, DISABLED_DumpSnapshots) {
  ::testing::Test::RecordProperty("backend", BackendId());
  auto base = LoadCanonicalEvents();
  std::vector<Event> ord = base; std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  const int K = 16;
  Pipeline p; p.K = K;
  std::size_t idx = 0;
  for (std::size_t i = 0; i < ord.size(); ++i) {
    p.Process(ord[i]); ++idx;
    if (idx == 16 || idx == 32 || idx == 48) {
      const std::string prefix = TestsDir() + "/golden/snapshot_" + std::to_string(idx);
      // Save Cluster
      {
        std::ofstream os(prefix + ".cluster.hser1", std::ios::binary);
        ASSERT_TRUE(SaveCluster(os, p.cmem));
      }
      // Save Prototype
      {
        std::ofstream os(prefix + ".prototype.hser1", std::ios::binary);
        ASSERT_TRUE(SavePrototype(os, p.pmem));
      }
      // Save sidecar state (last_obs words + mix) as tiny JSON
      {
        std::ofstream os(prefix + ".state.json", std::ios::binary);
        const auto& words = p.last_obs.Words();
        os << "{\"mix\":\"" << Hex64(p.mix) << "\",\"last_obs\":[";
        for (std::size_t w = 0; w < words.size(); ++w) {
          if (w) os << ","; char buf[19]; std::snprintf(buf, sizeof(buf), "\"0x%016llx\"", static_cast<unsigned long long>(words[w])); os << buf;
        }
        os << "]}";
      }
    }
  }
}

// Verify that resuming from snapshots reproduces the golden suffix of checkpoints and final hash
TEST(SnapshotRestore, Parity_MultipleSnapshotPoints) {
  ::testing::Test::RecordProperty("backend", BackendId());
  auto base = LoadCanonicalEvents();
  std::vector<Event> ord = base; std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  const int K = 16; const std::vector<int> points{16,32,48};

  // Compute uninterrupted golden suffixes from existing golden file
  const std::string json = ReadTextFile(TestsDir() + "/golden/streaming_hashes.json");
  std::vector<std::string> exp_chk_hex; std::string exp_final_hex;
  ASSERT_TRUE(ExtractBackendHashes(json, BackendId(), &exp_chk_hex, &exp_final_hex));
  ASSERT_GE(exp_chk_hex.size(), 4u); // K=16 over 64 events -> 4 checkpoints

  for (int N : points) {
    // Load snapshot fixtures

TEST(SnapshotRestore, DISABLED_DebugPrint) {
  auto base = LoadCanonicalEvents();
  std::vector<Event> ord = base; std::stable_sort(ord.begin(), ord.end(), TotalOrderLt);
  const int N = 16; const int K = 16;
  Pipeline p; p.K = K;
  const std::string prefix = TestsDir() + "/golden/snapshot_" + std::to_string(N);
  {
    std::ifstream is(prefix + ".cluster.hser1", std::ios::binary);
    ASSERT_TRUE(LoadCluster(is, &p.cmem));
  }
  {
    std::ifstream is(prefix + ".prototype.hser1", std::ios::binary);
    ASSERT_TRUE(LoadPrototype(is, &p.pmem));
  }
  {
    std::ifstream is(prefix + ".state.json", std::ios::binary);
    std::ostringstream ss; ss << is.rdbuf(); const std::string s = ss.str();
    auto mpos = s.find("\"mix\":\""); auto q1 = s.find('"', mpos + 7); auto q2 = s.find('"', q1 + 1);
    const std::string mix_hex = s.substr(q1 + 1, q2 - q1 - 1); p.mix = std::strtoull(mix_hex.c_str()+2, nullptr, 16);
    auto lpos = s.find("\"last_obs\""); auto a = s.find('[', lpos); auto b = s.find(']', a); std::size_t widx = 0; auto ppos = a + 1;
    while (ppos < b && widx < p.last_obs.Words().size()) { auto q = s.find('"', ppos); auto qn = s.find('"', q + 1); const std::string hex = s.substr(q + 1, qn - q - 1); p.last_obs.Words()[widx++] = std::strtoull(hex.c_str()+2, nullptr, 16); ppos = qn + 1; }
  }
  std::printf("loaded_hash@%d=%s\n", N, Hex64(p.CheckpointHash()).c_str());
}

    const std::string prefix = TestsDir() + "/golden/snapshot_" + std::to_string(N);
    Pipeline p; p.K = K;
    // Load Cluster
    {
      std::ifstream is(prefix + ".cluster.hser1", std::ios::binary);
      ASSERT_TRUE(is.good()) << "missing fixture: " << prefix << ".cluster.hser1";
      ASSERT_TRUE(LoadCluster(is, &p.cmem));
    }
    // Load Prototype
    {
      std::ifstream is(prefix + ".prototype.hser1", std::ios::binary);
      ASSERT_TRUE(is.good()) << "missing fixture: " << prefix << ".prototype.hser1";
      ASSERT_TRUE(LoadPrototype(is, &p.pmem));
    }
    // Load sidecar state
    {
      std::ifstream is(prefix + ".state.json", std::ios::binary);
      ASSERT_TRUE(is.good()) << "missing fixture: " << prefix << ".state.json";
      std::ostringstream ss; ss << is.rdbuf(); const std::string s = ss.str();
      // parse mix
      auto mpos = s.find("\"mix\":\""); ASSERT_NE(mpos, std::string::npos);
      auto q1 = s.find('"', mpos + 7); auto q2 = s.find('"', q1 + 1); ASSERT_NE(q1, std::string::npos); ASSERT_NE(q2, std::string::npos);
      const std::string mix_hex = s.substr(q1 + 1, q2 - q1 - 1);
      p.mix = std::strtoull(mix_hex.c_str()+2, nullptr, 16);
      // parse last_obs
      auto lpos = s.find("\"last_obs\""); ASSERT_NE(lpos, std::string::npos);
      auto a = s.find('[', lpos); auto b = s.find(']', a); ASSERT_NE(a, std::string::npos); ASSERT_NE(b, std::string::npos);
      std::size_t widx = 0; auto ppos = a + 1;
      while (ppos < b && widx < p.last_obs.Words().size()) {
        auto q = s.find('"', ppos); if (q == std::string::npos || q >= b) break; auto qn = s.find('"', q + 1); if (qn == std::string::npos || qn > b) break;
        const std::string hex = s.substr(q + 1, qn - q - 1);
        p.last_obs.Words()[widx++] = std::strtoull(hex.c_str()+2, nullptr, 16);
        ppos = qn + 1;
      }
    }

    // Resume from N+1
    std::vector<std::uint64_t> resumed_chk;
    for (std::size_t i = static_cast<std::size_t>(N); i < ord.size(); ++i) {
      p.Process(ord[i]);
      const std::size_t idx = i + 1;
      if ((idx % K) == 0) resumed_chk.push_back(p.CheckpointHash());
    }
    const std::uint64_t resumed_final = p.CheckpointHash();

    // Compare to golden suffix (positions > N)
    const std::size_t total_checkpoints = exp_chk_hex.size();
    const std::size_t first_chk_index = static_cast<std::size_t>((N / K)); // 1-based multiples; index is 0-based
    ASSERT_EQ(resumed_chk.size(), total_checkpoints - first_chk_index);
    for (std::size_t i = 0; i < resumed_chk.size(); ++i) {
      EXPECT_EQ(Hex64(resumed_chk[i]), exp_chk_hex[first_chk_index + i]) << "N=" << N << ", chk i=" << i;
    }
    EXPECT_EQ(Hex64(resumed_final), exp_final_hex) << "N=" << N;
  }
}

} // namespace

