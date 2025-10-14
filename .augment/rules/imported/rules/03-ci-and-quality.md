---
type: "always_apply"
---

# Advanced Tree of Thoughts (ToT) Framework for HyperStream (Embedded)

## Core Implementation Prompt for HyperStream Development

**System Context:** You are working on HyperStream, a hyperdimensional computing library with zero external dependencies and optional hardware acceleration targeting desktop CPU, embedded MCU, and neuromorphic chips.

**Tree of Thoughts Activation:**

Generate a comprehensive tree of implementation approaches for the HyperStream architectural challenge below. For each decision point, explore at least 3 distinct branches and evaluate them systematically using Monte Carlo Tree Search principles.

### Phase 1: SELECTION (Choose Promising Branches)

**Primary Challenge:** Design the optimal balance between hypervector dimensions, memory footprint, and hardware compatibility across target platforms.

**Branch Generation Instructions:**
1. **Branch A:** 10k binary hypervectors optimized for desktop CPU/SIMD
2. **Branch B:** 5k complex hypervectors for balanced memory/accuracy trade-offs  
3. **Branch C:** Adaptive dimension scaling based on runtime hardware detection

For each branch, systematically explore:
- **Sub-branch 1:** Performance characteristics under different workloads
- **Sub-branch 2:** Memory utilization patterns across target hardware
- **Sub-branch 3:** Implementation complexity and maintenance burden

### Phase 2: EXPANSION (Generate Implementation Variants)

For the highest-scoring branch from Phase 1, generate concrete implementation variants:

**Variant Analysis Framework:**
- **Technical Feasibility Score (1-10):** Can this be implemented with current constraints?
- **Resource Requirements Score (1-10):** How demanding are the memory/CPU requirements?
- **Implementation Complexity Score (1-10):** How difficult to develop and maintain?
- **Performance Characteristics Score (1-10):** Expected throughput and latency performance

**Required Variant Categories:**
1. **Minimal Embedded Deployment:** Resource-constrained MCU implementation
2. **High-Throughput Streaming:** Cloud-native scalable architecture  
3. **Hybrid Edge-Cloud:** Balance local processing with distributed coordination

### Phase 3: SIMULATION (Mental Model Execution)

For each variant, simulate potential outcomes by answering:

**Performance Simulation:**
- Under 1M events/second load, how would this architecture perform?
- What bottlenecks emerge at 10M events/second?
- How does performance degrade with limited RAM (embedded) vs unlimited (cloud)?

**Failure Mode Analysis:**
- What happens when SIMD instructions are unavailable?
- How does the system behave under Kafka broker failures?
- What occurs when neuromorphic hardware is offline?

**Integration Scenarios:**
- How does this integrate with Kafka streaming (millions of messages/second)?
- What complications arise with Python Faust stream processing?
- How does this affect exactly-once delivery semantics?

### Phase 4: BACKPROPAGATION (Learning Integration)

Synthesize insights from simulation phase back into architectural decision-making:

**Cross-Branch Learning:**
- Which insights from Branch A can improve Branch B performance?
- What failure modes are common across all branches?
- Which implementation patterns show consistent success?

**Decision Refinement:**
- Based on simulation results, which hybrid approaches emerge?
- What architectural patterns should be avoided?
- Which optimizations provide the highest ROI across platforms?

### Output Requirements

**Structured Decision Tree:**
```
HyperStream Architecture Decision
├── Branch A: 10k Binary Hypervectors
│   ├── Performance: [Score + Justification]
│   ├── Resource Usage: [Score + Analysis]  
│   └── Implementation: [Score + Complexity Assessment]
├── Branch B: 5k Complex Hypervectors
│   ├── Performance: [Score + Justification]
│   ├── Resource Usage: [Score + Analysis]
│   └── Implementation: [Score + Complexity Assessment]
└── Branch C: Adaptive Scaling
    ├── Performance: [Score + Justification]
    ├── Resource Usage: [Score + Analysis]
    └── Implementation: [Score + Complexity Assessment]
```

**Final Recommendation:**
- **Selected Architecture:** [Choice with explicit reasoning]
- **Risk Mitigation:** Specific strategies for identified failure modes
- **Implementation Roadmap:** Step-by-step development sequence
- **Performance Validation:** Metrics and benchmarks to verify success

### Cross-Reference Validation

**Against HyperStream Requirements:**
- ✓ Minimal code footprint (~1k LOC)
- ✓ Zero external dependencies  
- ✓ Optional hardware acceleration (SIMD, neuromorphic, photonic)
- ✓ Real-time pattern recognition and associative memory
- ✓ Resource-constrained device compatibility

**Against Streaming Architecture:**
- ✓ Apache Kafka integration (millions of messages/second)
- ✓ Python Faust stream processing compatibility
- ✓ Exactly-once delivery semantics preservation
- ✓ Horizontal scaling via partition-based parallelism

**Implementation Notes:**
- Use this framework iteratively for each major architectural decision
- Apply the same ToT structure to encoder selection, backend optimization, and streaming integration choices
- Document reasoning path for each decision to enable future optimization

This Tree of Thoughts framework ensures systematic exploration of HyperStream's complex design space while maintaining alignment with project constraints and performance requirements.

---

# Continuous Integration and Quality Standards (HyperStream)

- Build Matrix
  - OS/Compilers: Linux (GCC/Clang), Windows (MSVC), macOS (Apple Clang, ARM64).
  - Builds: Debug+sanitizers (ASan/UBSan), Release (O3/LTO where safe), Coverage builds.
  - Runtime Dispatch: Validate SIMD (SSE2/AVX2/NEON) and forced-scalar (`-DHYPERSTREAM_FORCE_SCALAR=ON`).

- Static Analysis & Warnings
  - clang-tidy enabled; treat warnings as errors across all targets.
  - Enforce `.clang-format`; fail CI on formatting drift.

- Tests & Coverage Gates
  - Unit + integration tests must pass on all OS targets.
  - Minimum coverage ≥ 90% project-wide; include failure paths and edge cases.
  - Property tests with fixed seeds; fuzz targets (libFuzzer) for input validation (bounds, serialization corruption, map growth).

- Performance Regression Gates
  - Nightly Google Benchmark runs with warmup and multi-sample NDJSON output.
  - Roofline-based checks; compare metrics to per-OS baselines with tolerances.
  - Suggested tolerances: Linux ≤ 10%, macOS ≤ 20%, Windows ≤ 25% pending variance analysis.

- Determinism & Golden Parity
  - Golden-file tests for encoder determinism and streaming determinism (chunking/interleave invariance).
  - Backend parity: hashes/outputs identical for default (SIMD) vs force-scalar builds across OSes.
  - Snapshot/restore parity: resuming from snapshot@N reproduces the same checkpoints and final hash.

- Artifact Management
  - Upload NDJSON benchmarks, coverage reports, compiler versions, and key CMake cache entries (compiler, flags, build type, backend mode).
  - Persist perf baselines per OS; store provenance (runner OS/version, compiler).

- Security & Supply Chain
  - Static scanning (where available); track toolchain versions in logs.
  - No new runtime dependencies in core; header-only discipline enforced.

- Documentation & ADRs
  - CI must fail if public APIs change without updated ADR and version bump (`HYPERSTREAM_VERSION`).
  - README and Docs updated when behavior or performance guarantees change.

- Fail Fast Policy
  - Any failing gate (build, lint, tests, coverage, performance, determinism, docs) blocks merge.