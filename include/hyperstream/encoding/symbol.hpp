#pragma once

// SymbolEncoder: compositional wrapper around ItemMemory providing
// ergonomic symbol/id encoding and optional role-based rotation.

#include <cstddef>
#include <cstdint>
#include <string_view>

#include "hyperstream/core/hypervector.hpp"
#include "hyperstream/core/ops.hpp"
#include "hyperstream/encoding/item_memory.hpp"

namespace hyperstream {
namespace encoding {

/**
 * @brief Symbol encoder built on ItemMemory with optional role permutation.
 *
 * @tparam Dim Hypervector dimension (bits)
 *
 * Thread-safety: stateless aside from construction-time seed; reentrant.
 * Complexity: O(Dim/64) per encode, plus O(Dim/64) for role rotation when role != 0.
 */
template <std::size_t Dim>
class SymbolEncoder {
 public:
  explicit SymbolEncoder(std::uint64_t seed) : im_(seed) {}

  void EncodeToken(std::string_view token, core::HyperVector<Dim, bool>* out) const {
    im_.EncodeToken(token, out);
  }

  void EncodeId(std::uint64_t id, core::HyperVector<Dim, bool>* out) const noexcept {
    im_.EncodeId(id, out);
  }

  // Encode token with role-based rotation by 'role' steps.
  void EncodeTokenRole(std::string_view token, std::size_t role,
                       core::HyperVector<Dim, bool>* out) const {
    if (role == 0) {
      im_.EncodeToken(token, out);
      return;
    }
    core::HyperVector<Dim, bool> base;
    im_.EncodeToken(token, &base);
    core::PermuteRotate(base, role, out);
  }

 private:
  ItemMemory<Dim> im_;
};

}  // namespace encoding
}  // namespace hyperstream

