#pragma once

// Minimal binary serialization for PrototypeMemory and ClusterMemory.
// Header-only; validates sizes; little-endian; no dynamic allocation.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include <iosfwd>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"

namespace hyperstream {
namespace io {

namespace detail_ser {
inline bool Write(std::ostream& os, const void* p, std::size_t n) {
  os.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
  return static_cast<bool>(os);
}
inline bool Read(std::istream& is, void* p, std::size_t n) {
  is.read(reinterpret_cast<char*>(p), static_cast<std::streamsize>(n));
  return static_cast<bool>(is);
}
}

enum class ObjectKind : std::uint8_t { Prototype = 1, Cluster = 2 };

struct Header {
  char magic[5];      // "HSER1"
  ObjectKind kind;    // 1=Prototype, 2=Cluster
  std::uint64_t dim;
  std::uint64_t capacity;
  std::uint64_t size;
};

inline Header MakeHeader(ObjectKind kind, std::uint64_t dim, std::uint64_t cap, std::uint64_t size) {
  Header h{};
  std::memcpy(h.magic, "HSER1", 5);
  h.kind = kind;
  h.dim = dim;
  h.capacity = cap;
  h.size = size;
  return h;
}

inline bool CheckMagic(const Header& h) {
  return std::memcmp(h.magic, "HSER1", 5) == 0;
}

/** Save PrototypeMemory to binary stream. */
template <std::size_t Dim, std::size_t Capacity>
bool SavePrototype(std::ostream& os, const memory::PrototypeMemory<Dim, Capacity>& mem) noexcept {
  using HV = core::HyperVector<Dim, bool>;
  using Entry = typename memory::PrototypeMemory<Dim, Capacity>::Entry;
  const auto* data = mem.data();
  const std::uint64_t size = static_cast<std::uint64_t>(mem.size());
  const Header h = MakeHeader(ObjectKind::Prototype, Dim, Capacity, size);
  if (!detail_ser::Write(os, &h, sizeof(h))) return false;
  const std::size_t word_count = HV::WordCount();
  for (std::size_t i = 0; i < mem.size(); ++i) {
    const Entry& e = data[i];
    if (!detail_ser::Write(os, &e.label, sizeof(e.label))) return false;
    const auto& words = e.hv.Words();
    if (!detail_ser::Write(os, words.data(), word_count * sizeof(std::uint64_t))) return false;
  }
  return true;
}

/** Load PrototypeMemory from binary stream. Precondition: mem->size() == 0. */
template <std::size_t Dim, std::size_t Capacity>
bool LoadPrototype(std::istream& is, memory::PrototypeMemory<Dim, Capacity>* mem) noexcept {
  if (mem == nullptr) return false;
  if (mem->size() != 0) return false;  // do not append; require empty
  Header h{};
  if (!detail_ser::Read(is, &h, sizeof(h))) return false;
  if (!CheckMagic(h) || h.kind != ObjectKind::Prototype) return false;
  if (h.dim != Dim || h.capacity != Capacity) return false;
  if (h.size > Capacity) return false;
  using HV = core::HyperVector<Dim, bool>;
  for (std::uint64_t i = 0; i < h.size; ++i) {
    std::uint64_t label = 0;
    if (!detail_ser::Read(is, &label, sizeof(label))) return false;
    HV hv;
    hv.Clear();
    auto& words = hv.Words();
    if (!detail_ser::Read(is, words.data(), HV::WordCount() * sizeof(std::uint64_t))) return false;
    if (!mem->Learn(label, hv)) return false;
  }
  return true;
}

/** Save ClusterMemory to binary stream. */
template <std::size_t Dim, std::size_t Capacity>
bool SaveCluster(std::ostream& os, const memory::ClusterMemory<Dim, Capacity>& mem) noexcept {
  const auto v = mem.view();
  const Header h = MakeHeader(ObjectKind::Cluster, Dim, Capacity, static_cast<std::uint64_t>(v.size));
  if (!detail_ser::Write(os, &h, sizeof(h))) return false;
  if (v.size == 0) return true;
  if (!detail_ser::Write(os, v.labels, sizeof(std::uint64_t) * v.size)) return false;
  if (!detail_ser::Write(os, v.counts, sizeof(int) * v.size)) return false;
  if (!detail_ser::Write(os, v.sums, sizeof(int) * v.size * Dim)) return false;
  return true;
}

/** Load ClusterMemory from binary stream. Precondition: mem->size() == 0. */
template <std::size_t Dim, std::size_t Capacity>
bool LoadCluster(std::istream& is, memory::ClusterMemory<Dim, Capacity>* mem) noexcept {
  if (mem == nullptr) return false;
  if (mem->size() != 0) return false;
  Header h{};
  if (!detail_ser::Read(is, &h, sizeof(h))) return false;
  if (!CheckMagic(h) || h.kind != ObjectKind::Cluster) return false;
  if (h.dim != Dim || h.capacity != Capacity) return false;
  if (h.size > Capacity) return false;
  const std::size_t n = static_cast<std::size_t>(h.size);
  // Temporary buffers on stack for safety; small n expected. If large, could chunk.
  std::vector<std::uint64_t> labels(n);
  std::vector<int> counts(n);
  std::vector<int> sums(n * Dim);
  if (n > 0) {
    if (!detail_ser::Read(is, labels.data(), sizeof(std::uint64_t) * n)) return false;
    if (!detail_ser::Read(is, counts.data(), sizeof(int) * n)) return false;
    if (!detail_ser::Read(is, sums.data(), sizeof(int) * n * Dim)) return false;
  }
  if (!mem->LoadRaw(labels.data(), counts.data(), sums.data(), n)) return false;
  return true;
}

}  // namespace io
}  // namespace hyperstream

