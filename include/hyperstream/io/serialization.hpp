#pragma once

// HSER1 serialization: minimal, header-only, deterministic. v1.1 adds optional
// integrity trailer (tag+CRC32) while preserving backward-compatible loading of
// v1 payloads. Little-endian; no external dependencies.

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
// CRC32 (IEEE 802.3, polynomial 0xEDB88320), byte-wise, no table.
inline std::uint32_t Crc32(const std::uint8_t* data, std::size_t n) noexcept {
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < n; ++i) {
    crc ^= static_cast<std::uint32_t>(data[i]);
    for (int k = 0; k < 8; ++k) {
      if (crc & 1u) crc = (crc >> 1) ^ 0xEDB88320u; else crc >>= 1;
    }
  }
  return crc ^ 0xFFFFFFFFu;
}
inline void Crc32Update(std::uint32_t* crc, const void* p, std::size_t n) noexcept {
  const auto* b = reinterpret_cast<const std::uint8_t*>(p);
  std::uint32_t c = *crc;
  for (std::size_t i = 0; i < n; ++i) {
    c ^= static_cast<std::uint32_t>(b[i]);
    for (int k = 0; k < 8; ++k) {
      if (c & 1u) c = (c >> 1) ^ 0xEDB88320u; else c >>= 1;
    }
  }
  *crc = c;
}
inline bool WriteTrailer(std::ostream& os, std::uint32_t crc) {
  static constexpr char kTag[4] = {'H','S','X','1'}; // trailer tag
  return Write(os, kTag, 4) && Write(os, &crc, sizeof(crc));
}
inline bool TryReadTrailer(std::istream& is, std::uint32_t* out_crc) {
  // Only attempt when seekable to avoid consuming bytes on non-seekable streams
  const auto pos = is.tellg();
  if (pos == static_cast<std::streampos>(-1)) return false;
  char tag[4];
  if (!Read(is, tag, 4)) { is.clear(); is.seekg(pos); return false; }
  if (!(tag[0]=='H' && tag[1]=='S' && tag[2]=='X' && tag[3]=='1')) {
    // Not a trailer; rewind and treat as v1
    is.clear(); is.seekg(pos);
    return false;
  }
  std::uint32_t crc = 0;
  if (!Read(is, &crc, sizeof(crc))) { is.clear(); is.seekg(pos); return false; }
  if (out_crc) *out_crc = crc;
  return true;
}
} // namespace detail_ser

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

inline bool CheckMagic(const Header& h) { return std::memcmp(h.magic, "HSER1", 5) == 0; }

/** Save PrototypeMemory to binary stream. v1.1: append trailer tag+CRC32(payload). */
template <std::size_t Dim, std::size_t Capacity>
bool SavePrototype(std::ostream& os, const memory::PrototypeMemory<Dim, Capacity>& mem) noexcept {
  using HV = core::HyperVector<Dim, bool>;
  using Entry = typename memory::PrototypeMemory<Dim, Capacity>::Entry;
  const auto* data = mem.data();
  const std::uint64_t size = static_cast<std::uint64_t>(mem.size());
  const Header h = MakeHeader(ObjectKind::Prototype, Dim, Capacity, size);
  if (!detail_ser::Write(os, &h, sizeof(h))) return false;
  const std::size_t word_count = HV::WordCount();
  std::uint32_t crc = 0xFFFFFFFFu;
  for (std::size_t i = 0; i < mem.size(); ++i) {
    const Entry& e = data[i];
    // Update CRC over payload bytes (label + hv words) and write
    detail_ser::Crc32Update(&crc, &e.label, sizeof(e.label));
    if (!detail_ser::Write(os, &e.label, sizeof(e.label))) return false;
    const auto& words = e.hv.Words();
    detail_ser::Crc32Update(&crc, words.data(), word_count * sizeof(std::uint64_t));
    if (!detail_ser::Write(os, words.data(), word_count * sizeof(std::uint64_t))) return false;
  }
#ifndef HYPERSTREAM_HSER1_WRITE_V1
  crc ^= 0xFFFFFFFFu;  // finalize
  if (!detail_ser::WriteTrailer(os, crc)) return false;
#endif
  return true;
}

/** Load PrototypeMemory from binary stream. Precondition: mem->size() == 0. */
template <std::size_t Dim, std::size_t Capacity>
bool LoadPrototype(std::istream& is, memory::PrototypeMemory<Dim, Capacity>* mem) noexcept {
  if (mem == nullptr) return false;
  if (mem->size() != 0) return false;  // require empty
  Header h{};
  if (!detail_ser::Read(is, &h, sizeof(h))) return false;
  if (!CheckMagic(h) || h.kind != ObjectKind::Prototype) return false;
  if (h.dim != Dim || h.capacity != Capacity) return false;
  if (h.size > Capacity) return false;
  using HV = core::HyperVector<Dim, bool>;
  std::uint32_t crc_calc = 0xFFFFFFFFu;
  for (std::uint64_t i = 0; i < h.size; ++i) {
    std::uint64_t label = 0;
    if (!detail_ser::Read(is, &label, sizeof(label))) return false;
    detail_ser::Crc32Update(&crc_calc, &label, sizeof(label));
    HV hv; hv.Clear();
    auto& words = hv.Words();
    if (!detail_ser::Read(is, words.data(), HV::WordCount() * sizeof(std::uint64_t))) return false;
    detail_ser::Crc32Update(&crc_calc, words.data(), HV::WordCount() * sizeof(std::uint64_t));
    if (!mem->Learn(label, hv)) return false;
  }
  // Optional trailer validation (v1.1). If present, must validate; if absent, accept as v1.
  std::uint32_t crc_file = 0;
  if (detail_ser::TryReadTrailer(is, &crc_file)) {
    crc_calc ^= 0xFFFFFFFFu;
    if (crc_calc != crc_file) return false;
  }
  return true;
}

/** Save ClusterMemory to binary stream. v1.1: append trailer tag+CRC32(payload). */
template <std::size_t Dim, std::size_t Capacity>
bool SaveCluster(std::ostream& os, const memory::ClusterMemory<Dim, Capacity>& mem) noexcept {
  const auto v = mem.view();
  const Header h = MakeHeader(ObjectKind::Cluster, Dim, Capacity, static_cast<std::uint64_t>(v.size));
  if (!detail_ser::Write(os, &h, sizeof(h))) return false;
  std::uint32_t crc = 0xFFFFFFFFu;
  if (v.size == 0) {
#ifndef HYPERSTREAM_HSER1_WRITE_V1
    crc ^= 0xFFFFFFFFu;
    if (!detail_ser::WriteTrailer(os, crc)) return false;
#endif
    return true;
  }
  detail_ser::Crc32Update(&crc, v.labels, sizeof(std::uint64_t) * v.size);
  if (!detail_ser::Write(os, v.labels, sizeof(std::uint64_t) * v.size)) return false;
  detail_ser::Crc32Update(&crc, v.counts, sizeof(int) * v.size);
  if (!detail_ser::Write(os, v.counts, sizeof(int) * v.size)) return false;
  detail_ser::Crc32Update(&crc, v.sums, sizeof(int) * v.size * Dim);
  if (!detail_ser::Write(os, v.sums, sizeof(int) * v.size * Dim)) return false;
#ifndef HYPERSTREAM_HSER1_WRITE_V1
  crc ^= 0xFFFFFFFFu;
  if (!detail_ser::WriteTrailer(os, crc)) return false;
#endif
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
  std::vector<std::uint64_t> labels(n);
  std::vector<int> counts(n);
  std::vector<int> sums(n * Dim);
  std::uint32_t crc_calc = 0xFFFFFFFFu;
  if (n > 0) {
    if (!detail_ser::Read(is, labels.data(), sizeof(std::uint64_t) * n)) return false;
    detail_ser::Crc32Update(&crc_calc, labels.data(), sizeof(std::uint64_t) * n);
    if (!detail_ser::Read(is, counts.data(), sizeof(int) * n)) return false;
    detail_ser::Crc32Update(&crc_calc, counts.data(), sizeof(int) * n);
    if (!detail_ser::Read(is, sums.data(), sizeof(int) * n * Dim)) return false;
    detail_ser::Crc32Update(&crc_calc, sums.data(), sizeof(int) * n * Dim);
  }
  if (!mem->LoadRaw(labels.data(), counts.data(), sums.data(), n)) return false;
  std::uint32_t crc_file = 0;
  if (detail_ser::TryReadTrailer(is, &crc_file)) {
    crc_calc ^= 0xFFFFFFFFu;
    if (crc_calc != crc_file) return false;
  }
  return true;
}

}  // namespace io
}  // namespace hyperstream
