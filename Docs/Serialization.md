# Serialization

Header-only, minimal binary serialization for associative memories.

- Namespace: `hyperstream::io`
- Header: `include/hyperstream/io/serialization.hpp`
- Scope: `PrototypeMemory<Dim,Capacity>` and `ClusterMemory<Dim,Capacity>`
- Guarantees: deterministic, versioned header, bounds-checked, no exceptions (bool return)

## Format

Little-endian binary layout with a fixed header followed by payload.

Header (struct):
- magic[5] = "HSER1"
- kind: uint8 (1 = Prototype, 2 = Cluster)
- dim:  uint64
- capacity: uint64
- size: uint64 (number of valid entries/clusters)

## Versioning and Integrity

- Versions:
  - v1: Header + payload only
  - v1.1: Header + payload + trailer (tag + CRC32 over payload)
- Trailer:
  - Tag: ASCII "HSX1" (4 bytes), followed by CRC32 (IEEE 802.3, 4 bytes, little-endian)
  - Writer defaults to v1.1; define HYPERSTREAM_HSER1_WRITE_V1 to emit v1 (no trailer)
- Reader behavior:
  - Always reads v1 payloads
  - If the stream is seekable and the trailer tag is present after payload, validates CRC32; mismatch causes load to fail
  - On non-seekable streams, CRC validation may be skipped (treated as v1)
- Backward-compatibility:
  - Existing v1 artifacts continue to load unchanged
  - v1.1 artifacts append a trailer; legacy v1 loaders that stop after payload will ignore trailer bytes

Payloads:

- Prototype:
  - Repeat for i in [0, size):
    - label: uint64
    - hv: `HyperVector<Dim,bool>` words (WordCount() * uint64)
- Cluster:
  - labels[ size ]: uint64
  - counts[ size ]: int32
  - sums[ size * Dim ]: int32 (row-major per cluster)

## API

```c++
bool SavePrototype(std::ostream&, const PrototypeMemory<Dim,Cap>&) noexcept;
bool LoadPrototype(std::istream&, PrototypeMemory<Dim,Cap>*) noexcept;  // requires empty mem

bool SaveCluster(std::ostream&, const ClusterMemory<Dim,Cap>&) noexcept;
bool LoadCluster(std::istream&, ClusterMemory<Dim,Cap>*) noexcept;      // requires empty mem
```

- Returns true on success, false on any validation or I/O failure
- Load preconditions: target memory must be empty (`size()==0`)

## Validation and safety

- Magic + kind + invariants (dim, capacity) must match the template instance
- `size <= Capacity` checked before reading payload
- For Prototype: incremental "learn" after reading each entry
- For Cluster: internal arrays loaded atomically via `ClusterMemory::LoadRaw`
- No dynamic allocations in `Save*`; `LoadCluster` uses bounded temporary vectors

## Threading

- APIs are not thread-safe; use external synchronization if sharing streams/memories

## Compatibility

- Little-endian only (current targets: x86_64, AArch64). Cross-endian is not supported.
- Struct packing is avoided by using explicit field writes/reads.

## Examples

See `tests/serialization_tests.cc` for round-trip and corruption-detection tests.

