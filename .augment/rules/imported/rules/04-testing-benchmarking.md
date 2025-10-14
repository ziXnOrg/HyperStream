---
type: "always_apply"
---

# Testing & Benchmarking (HyperStream)

- Unit Tests
  - Use Google Test; cover core ops (bind/bundle/permute/similarity), encoders, memories.
  - Property tests (invertibility, triangle inequality, permutation composition) with fixed seeds.

- Fuzz Testing
  - libFuzzer targets for input validation paths (bounds, serialization corruption, map growth).

- Benchmarks
  - Warmup phase, multiple samples, NDJSON output; pin CPU affinity where possible; disable turbo when supported.
  - Include bind, hamming, permute, associative memory classify/update.

- Acceptance Targets
  - Coverage ≥ 90%; AM classify ≥ target QPS; bind/hamming GB/s within baseline tolerances.

- Edge/MCU Testing
  - No dynamic allocation; static memory budgets enforced; latency and footprint recorded per dimension.