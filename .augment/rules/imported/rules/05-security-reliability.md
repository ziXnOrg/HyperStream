---
type: "always_apply"
---

# Security & Reliability (HyperStream)

- Input Validation
  - Bounds checks on all external inputs; validate dimensions/word counts; fail closed.

- Serialization
  - HSER1 format with CRC; strict little-endian; deterministic I/O; corruption returns explicit status codes.

- ABI/FFI Safety
  - C API uses opaque handles and status codes; no exceptions cross the boundary.
  - Thread safety: not thread-safe by default; caller must synchronize shared handles.

- Supply Chain & Provenance
  - Pin toolchain versions in CI; record compiler, flags, backend mode in artifacts.

- Streaming & Data (when used)
  - TLS for transport, authn/z, exactly-once semantics; versioned snapshots for state.