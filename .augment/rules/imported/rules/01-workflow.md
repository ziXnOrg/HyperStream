---
type: "always_apply"
---

# Advanced Tree of Thoughts (ToT) Framework for HyperStream (Embedded)

## Core Implementation Prompt for HyperStream Development

**System Context:** You are working on HyperStream, a minimal (~1,000 LOC) hyperdimensional computing library with zero external dependencies and optional hardware acceleration targeting desktop CPU, embedded MCU, and neuromorphic chips.

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

# Workflow Guidelines (HyperStream)

- Branching
  - Create short-lived feature branches off `main` (e.g., `feat/phase-x-topic`).
  - Keep changes atomic and reviewable; prefer multiple small PRs over one large PR.

- Decision-making
  - Follow the embedded ToT workflow above for all non-trivial decisions.
  - For public API/behavior changes, add an ADR in `Docs/ADR/adr_XXXX_title.md` and reference it in the PR.

- Tracking
  - Use GitHub Issues; label by Phase (A..H) and component (`core`, `backend`, `encoding`, `memory`, `ci`, `docs`).
  - Cross-link PRs ↔ Issues ↔ ADRs.

- Review gates (must pass before merge)
  - Build and tests green on linux/windows/macos.
  - Coverage ≥ 90% total; include failure paths.
  - Perf regression checks pass (bind/hamming/AM) vs baselines.
  - Lint clean (clang-tidy/format; warnings-as-errors).

- Releases & compatibility
  - Semantic versioning; update `HYPERSTREAM_VERSION` and changelog.
  - Deprecations documented; maintain backward compatibility on public headers.

- Core constraints
  - Core remains header-only; zero new runtime dependencies.
  - Determinism: fixed RNG seeds by default; env override for stress runs.