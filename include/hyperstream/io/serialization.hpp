#pragma once

// =============================================================================
// File:        include/hyperstream/io/serialization.hpp
// Overview:    HSER1 binary serialization for HyperStream memories (v1/v1.1).
//              v1.1 adds optional integrity trailer (tag + CRC32) while keeping
//              v1 readers compatible.
// Mathematical Foundation: None specific; CRC32 IEEE 802.3 polynomial used for
//              trailer integrity verification.
// Security Considerations: Validates header/magic; bounds check sizes; optional
//              trailer verification; does not throw in hot paths by default.
// Performance Considerations: Single-pass streaming; byte-wise CRC32 without
//              tables to avoid large static data; use of std::array where fixed.
// Examples:    SavePrototype/LoadPrototype and SaveCluster/LoadCluster helpers.
// =============================================================================
// HSER1 serialization: minimal, header-only, deterministic. v1.1 adds optional
// integrity trailer (tag+CRC32) while preserving backward-compatible loading of
// v1 payloads. Little-endian; no external dependencies.

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <string_view>
#include <vector>
#include <array> // Added for std::array

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/memory/associative.hpp"

namespace hyperstream::io {

namespace detail_ser {
inline bool Write(std::ostream& output_stream, const void* buffer_ptr, std::size_t byte_count) {
  output_stream.write(static_cast<const char*>(buffer_ptr), static_cast<std::streamsize>(byte_count));
  return static_cast<bool>(output_stream);
}
inline bool Read(std::istream& input_stream, void* buffer_ptr, std::size_t byte_count) {
  input_stream.read(static_cast<char*>(buffer_ptr), static_cast<std::streamsize>(byte_count));
  return static_cast<bool>(input_stream);
}
// CRC32 (IEEE 802.3, polynomial 0xEDB88320), byte-wise, no table.
inline std::uint32_t Crc32(const void* buffer_ptr, std::size_t byte_count) noexcept {
  static constexpr std::uint32_t kCrcInit = 0xFFFFFFFFU;
  static constexpr std::uint32_t kCrcPoly = 0xEDB88320U;
  static constexpr int kBitsPerByte = 8;
  std::uint32_t crc_value = kCrcInit;
  const auto* bytes = static_cast<const char*>(buffer_ptr);
  for (unsigned char byte_value : std::string_view(bytes, byte_count)) {
    crc_value ^= static_cast<std::uint32_t>(static_cast<std::uint8_t>(byte_value));
    for (int bit_index = 0; bit_index < kBitsPerByte; ++bit_index) {
      if ((crc_value & 1U) != 0U) {
        crc_value = (crc_value >> 1) ^ kCrcPoly;
      } else {
        crc_value >>= 1;
      }
    }
  }
  return crc_value ^ kCrcInit;
}
inline void Crc32Update(std::uint32_t* crc, const void* buffer_ptr, std::size_t byte_count) noexcept {
  static constexpr std::uint32_t kCrcPoly = 0xEDB88320U;
  static constexpr int kBitsPerByte = 8;
  std::uint32_t crc_value = *crc;
  const auto* bytes = static_cast<const char*>(buffer_ptr);
  for (unsigned char byte_value : std::string_view(bytes, byte_count)) {
    crc_value ^= static_cast<std::uint32_t>(static_cast<std::uint8_t>(byte_value));
    for (int bit_index = 0; bit_index < kBitsPerByte; ++bit_index) {
      if ((crc_value & 1U) != 0U) {
        crc_value = (crc_value >> 1) ^ kCrcPoly;
      } else {
        crc_value >>= 1;
      }
    }
  }
  *crc = crc_value;
}
inline bool WriteTrailer(std::ostream& output_stream, std::uint32_t crc) {
  static constexpr std::size_t kTrailerTagSize = 4;
  static constexpr std::array<char, kTrailerTagSize> kTag = {'H', 'S', 'X', '1'};  // trailer tag
  return Write(output_stream, kTag.data(), kTag.size()) && Write(output_stream, &crc, sizeof(crc));
}
inline bool TryReadTrailer(std::istream& input_stream, std::uint32_t* out_crc) {
  // Only attempt when seekable to avoid consuming bytes on non-seekable streams
  const auto pos = input_stream.tellg();
  if (pos == static_cast<std::streampos>(-1)) {
    return false;
  }
  static constexpr std::size_t kTrailerTagSize = 4;
  std::array<char, kTrailerTagSize> tag{};
  if (!Read(input_stream, tag.data(), tag.size())) {
    input_stream.clear();
    input_stream.seekg(pos);
    return false;
  }
  if (tag[0] != 'H' || tag[1] != 'S' || tag[2] != 'X' || tag[3] != '1') {
    // Not a trailer; rewind and treat as v1
    input_stream.clear();
    input_stream.seekg(pos);
    return false;
  }
  std::uint32_t crc = 0;
  if (!Read(input_stream, &crc, sizeof(crc))) {
    input_stream.clear();
    input_stream.seekg(pos);
    return false;
  }
  if (out_crc != nullptr) {
    *out_crc = crc;
  }
  return true;
}
}  // namespace detail_ser

enum class ObjectKind : std::uint8_t { Prototype = 1, Cluster = 2 };

struct Header {
  static constexpr std::size_t kMagicSize = 5;
  std::array<char, kMagicSize> magic;    // "HSER1"
  ObjectKind kind;  // 1=Prototype, 2=Cluster
  std::uint64_t dim;
  std::uint64_t capacity;
  std::uint64_t size;
};

struct HeaderInputs {
  ObjectKind kind;
  std::uint64_t dim_bits;
  std::uint64_t capacity;
  std::uint64_t num_items;
};

inline Header MakeHeader(const HeaderInputs& inputs) {
  Header header{};
  std::memcpy(header.magic.data(), "HSER1", Header::kMagicSize);
  header.kind = inputs.kind;
  header.dim = inputs.dim_bits;
  header.capacity = inputs.capacity;
  header.size = inputs.num_items;
  return header;
}

inline bool CheckMagic(const Header& header) {
  static constexpr std::size_t kMagicSize = Header::kMagicSize;
  return std::memcmp(header.magic.data(), "HSER1", kMagicSize) == 0;
}

/** Save PrototypeMemory to binary stream. v1.1: append trailer tag+CRC32(payload). */
template <std::size_t Dim, std::size_t Capacity>
bool SavePrototype(std::ostream& output_stream, const memory::PrototypeMemory<Dim, Capacity>& mem) noexcept {
  using HV = core::HyperVector<Dim, bool>;
  using Entry = typename memory::PrototypeMemory<Dim, Capacity>::Entry;
  const auto* data = mem.Data();
  const std::uint64_t size = static_cast<std::uint64_t>(mem.Size());
  const Header h = MakeHeader(HeaderInputs{ObjectKind::Prototype, Dim, Capacity, size});
  if (!detail_ser::Write(output_stream, &h, sizeof(h))) return false;
  const std::size_t word_count = HV::WordCount();
  std::uint32_t crc = 0xFFFFFFFFU;
  for (std::size_t i = 0; i < mem.Size(); ++i) {
    const Entry& e = data[i];
    // Update CRC over payload bytes (label + hv words) and write
    detail_ser::Crc32Update(&crc, &e.label, sizeof(e.label));
    if (!detail_ser::Write(output_stream, &e.label, sizeof(e.label))) return false;
    const auto& words = e.hv.Words();
    detail_ser::Crc32Update(&crc, words.data(), word_count * sizeof(std::uint64_t));
    if (!detail_ser::Write(output_stream, words.data(), word_count * sizeof(std::uint64_t))) return false;
  }
#ifndef HYPERSTREAM_HSER1_WRITE_V1
  crc ^= 0xFFFFFFFFU;  // finalize
  if (!detail_ser::WriteTrailer(output_stream, crc)) return false;
#endif
  return true;
}

/** Load PrototypeMemory from binary stream. Precondition: mem->size() == 0. */
template <std::size_t Dim, std::size_t Capacity>
bool LoadPrototype(std::istream& input_stream, memory::PrototypeMemory<Dim, Capacity>* mem) noexcept {
  if (mem == nullptr) return false;
  if (mem->Size() != 0) return false;  // require empty
  Header h{};
  if (!detail_ser::Read(input_stream, &h, sizeof(h))) return false;
  if (!CheckMagic(h) || h.kind != ObjectKind::Prototype) return false;
  if (h.dim != Dim || h.capacity != Capacity) return false;
  if (h.size > Capacity) return false;
  using HV = core::HyperVector<Dim, bool>;
  std::uint32_t crc_calc = 0xFFFFFFFFU;
  for (std::uint64_t i = 0; i < h.size; ++i) {
    std::uint64_t label = 0;
    if (!detail_ser::Read(input_stream, &label, sizeof(label))) return false;
    detail_ser::Crc32Update(&crc_calc, &label, sizeof(label));
    HV hv;
    hv.Clear();
    auto& words = hv.Words();
    if (!detail_ser::Read(input_stream, words.data(), HV::WordCount() * sizeof(std::uint64_t))) return false;
    detail_ser::Crc32Update(&crc_calc, words.data(), HV::WordCount() * sizeof(std::uint64_t));
    if (!mem->Learn(label, hv)) return false;
  }
  // Optional trailer validation (v1.1). If present, must validate; if absent, accept as v1.
  std::uint32_t crc_file = 0;
  if (detail_ser::TryReadTrailer(input_stream, &crc_file)) {
    crc_calc ^= 0xFFFFFFFFU;
    if (crc_calc != crc_file) return false;
  }
  return true;
}

/** Save ClusterMemory to binary stream. v1.1: append trailer tag+CRC32(payload). */
template <std::size_t Dim, std::size_t Capacity>
bool SaveCluster(std::ostream& output_stream, const memory::ClusterMemory<Dim, Capacity>& mem) noexcept {
  const auto v = mem.GetView();
  const Header h = MakeHeader(HeaderInputs{ObjectKind::Cluster, Dim, Capacity, static_cast<std::uint64_t>(v.size)});
  if (!detail_ser::Write(output_stream, &h, sizeof(h))) return false;
  std::uint32_t crc = 0xFFFFFFFFU;
  if (v.size == 0) {
#ifndef HYPERSTREAM_HSER1_WRITE_V1
    crc ^= 0xFFFFFFFFU;
    if (!detail_ser::WriteTrailer(output_stream, crc)) return false;
#endif
    return true;
  }
  detail_ser::Crc32Update(&crc, v.labels, sizeof(std::uint64_t) * v.size);
  if (!detail_ser::Write(output_stream, v.labels, sizeof(std::uint64_t) * v.size)) return false;
  detail_ser::Crc32Update(&crc, v.counts, sizeof(int) * v.size);
  if (!detail_ser::Write(output_stream, v.counts, sizeof(int) * v.size)) return false;
  detail_ser::Crc32Update(&crc, v.sums, sizeof(int) * v.size * Dim);
  if (!detail_ser::Write(output_stream, v.sums, sizeof(int) * v.size * Dim)) return false;
#ifndef HYPERSTREAM_HSER1_WRITE_V1
  crc ^= 0xFFFFFFFFU;
  if (!detail_ser::WriteTrailer(output_stream, crc)) return false;
#endif
  return true;
}

/** Load ClusterMemory from binary stream. Precondition: mem->size() == 0. */
template <std::size_t Dim, std::size_t Capacity>
bool LoadCluster(std::istream& input_stream, memory::ClusterMemory<Dim, Capacity>* mem) noexcept {
  if (mem == nullptr) return false;
  if (mem->Size() != 0) return false;
  Header h{};
  if (!detail_ser::Read(input_stream, &h, sizeof(h))) return false;
  if (!CheckMagic(h) || h.kind != ObjectKind::Cluster) return false;
  if (h.dim != Dim || h.capacity != Capacity) return false;
  if (h.size > Capacity) return false;
  const std::size_t n = static_cast<std::size_t>(h.size);
  std::vector<std::uint64_t> labels(n);
  std::vector<int> counts(n);
  std::vector<int> sums(n * Dim);
  std::uint32_t crc_calc = 0xFFFFFFFFU;
  if (n > 0) {
    if (!detail_ser::Read(input_stream, labels.data(), sizeof(std::uint64_t) * n)) return false;
    detail_ser::Crc32Update(&crc_calc, labels.data(), sizeof(std::uint64_t) * n);
    if (!detail_ser::Read(input_stream, counts.data(), sizeof(int) * n)) return false;
    detail_ser::Crc32Update(&crc_calc, counts.data(), sizeof(int) * n);
    if (!detail_ser::Read(input_stream, sums.data(), sizeof(int) * n * Dim)) return false;
    detail_ser::Crc32Update(&crc_calc, sums.data(), sizeof(int) * n * Dim);
  }
  if (!mem->LoadRaw({labels.data(), counts.data(), sums.data(), n})) return false;
  std::uint32_t crc_file = 0;
  if (detail_ser::TryReadTrailer(input_stream, &crc_file)) {
    crc_calc ^= 0xFFFFFFFFU;
    if (crc_calc != crc_file) return false;
  }
  return true;
}

}  // namespace hyperstream::io
