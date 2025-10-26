# HyperStream Implementation Plan

## Implementation Status & Validation

### Current State Analysis
- **Completed**:
  - Core HyperVector (bool + typed) with bit-packing and word-wise permutation
  - SIMD backends (AVX2/SSE2/NEON) for bind/hamming; runtime dispatch & policy
  - Encoding: ItemMemory, SymbolEncoder, Thermometer, RandomProjection
  - Associative memory: Prototype, Cluster, Cleanup with serialization (HSER1)
  - HSER1 serialization v1/v1.1 with optional CRC trailer
  - Per-file clang-tidy cleanup; C++17 nested namespaces; standardized module headers
  - Tests: property-based, backend equality/determinism, snapshot/restore
  - Benchmarks: bind/hamming/permute/AM/cluster; NDJSON output
  - CI: build/test matrix, perf regression gates, golden determinism, backend parity

- **In Progress**:
  - Performance tuning (AVX2/SSE2 thresholds; further AVX2 tuning A/B)
  - Documentation/ADRs (policy thresholds; memory layout invariants; Batch 3 API updates recorded in ADR-0001)
  - Coverage increase toward ≥90%

### Validation Against Requirements
| Requirement | Status | Notes |
|-------------|--------|-------|
|<10k LOC core | ✅ On track | Header-only core remains compact |
| No external deps | ✅ Maintained | STL + intrinsics only |
| Cross-platform | ✅ Achieved | Windows/MSVC, Linux/GCC/Clang, macOS/ARM64 |
| Real-time capable | 🔄 In progress | Perf benches integrated; tuning ongoing |
| Test coverage ≥ 90% | 🔄 In progress | Trending upward; add more unit/prop tests |
| Performance benchmarks | ✅ In place | NDJSON harness; perf regression CI |
| Critical path tests | ✅ In place | Backend parity, determinism, snapshots |

## Test Implementation Plan

### Unit Tests for Critical Paths

#### 1.1.1 BitPackingCorrectness
- **Test Case**: Verify that bit packing and unpacking produce the same result
- **Test Code**:
#### 1.1 HyperVector Core Operations
```cpp
TEST(HyperVectorTest, BitPackingCorrectness) {
    // Test bit packing/unpacking
    constexpr size_t kDim = 1024;
    BinaryHyperVector hv(kDim);
    
    // Set alternating bits
    for (size_t i = 0; i < kDim; ++i) {
        hv.set_bit(i, i % 2 == 0);
    }
    
    // Verify bits
    for (size_t i = 0; i < kDim; ++i) {
        EXPECT_EQ(hv.get_bit(i), i % 2 == 0) << "Bit " << i << " mismatch";
    }
    
    // Test edge cases
    EXPECT_THROW(hv.get_bit(kDim + 1), std::out_of_range);
}

TEST(HyperVectorTest, SIMDConsistency) {
    // Verify SIMD and scalar implementations produce same results
    constexpr size_t kDim = 256;
    BinaryHyperVector a(kDim), b(kDim);
    
    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::bernoulli_distribution dist(0.5);
    
    for (size_t i = 0; i < kDim; ++i) {
        bool val = dist(gen);
        a.set_bit(i, val);
        b.set_bit(i, !val);
    }
    
    // Compare SIMD and scalar results
    BinaryHyperVector simd_result(kDim);
    BinaryHyperVector scalar_result(kDim);
    
    // Bind operation
    BindSIMD(a, b, &simd_result);
    BindScalar(a, b, &scalar_result);
    EXPECT_EQ(simd_result, scalar_result);
    
    // Hamming distance
    EXPECT_EQ(HammingDistanceSIMD(a, b), HammingDistanceScalar(a, b));
}
```

### Performance Benchmarks

#### 1.2 Benchmark Suite
```cpp
// Google Benchmark setup
static void BM_BindOperation(benchmark::State& state) {
    const size_t dim = state.range(0);
    BinaryHyperVector a(dim), b(dim), result(dim);
    
    // Warm-up
    BindSIMD(a, b, &result);
    
    // Benchmark loop
    for (auto _ : state) {
        BindSIMD(a, b, &result);
        benchmark::DoNotOptimize(result);
    }
    
    state.SetBytesProcessed(int64_t(state.iterations()) * dim / 8);
    state.SetLabel("Bind Operation");
}

// Register benchmarks with different sizes
BENCHMARK(BM_BindOperation)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20)  // 1K to 1M dimensions
    ->Unit(benchmark::kMicrosecond);

// Memory bandwidth benchmark
static void BM_MemoryBandwidth(benchmark::State& state) {
    const size_t size = state.range(0);
    std::vector<uint64_t> data(size);
    std::iota(data.begin(), data.end(), 0);
    
    uint64_t sum = 0;
    for (auto _ : state) {
        sum += std::accumulate(data.begin(), data.end(), 0ULL);
        benchmark::DoNotOptimize(sum);
    }
    
    state.SetBytesProcessed(int64_t(state.iterations()) * size * sizeof(uint64_t));
    state.SetLabel("Memory Bandwidth");
}

BENCHMARK(BM_MemoryBandwidth)->Range(1 << 10, 1 << 26);
```

### Test Coverage Requirements

#### 1.3 Coverage Goals
- **Unit Tests**: 100% line coverage for core operations
- **Integration Tests**: All public APIs with various input sizes
- **Fuzz Testing**: For all input validation code paths
- **Performance Tests**: Compare against theoretical maximums

#### 1.4 Test Matrix
| Component | Test Type | Target Coverage | Status |
|-----------|-----------|-----------------|--------|
| Core Ops | Unit | 100% | 🔄 85% |
| Encoders  | Unit/Int. | 95% | 🔄 70% |
| Memory    | Perf. Test | 90% | 🔄 60% |
| SIMD Ops  | Correctness | 100% | 🔄 90% |

## 1. Core Library Implementation

### 1.1 HyperVector Class

#### 1.1.1 Bit-Packed Binary HyperVectors
- **Implementation**:
  ```cpp
  // Layout: Pack 64 bits per uint64_t
  class BinaryHyperVector {
    std::vector<uint64_t> data_;
    static constexpr size_t BITS_PER_WORD = 64;
    
    // Bit manipulation utilities
    static size_t word_index(size_t bit_pos) { return bit_pos / BITS_PER_WORD; }
    static size_t bit_offset(size_t bit_pos) { return bit_pos % BITS_PER_WORD; }
    
  public:
    explicit BinaryHyperVector(size_t dim) : data_((dim + BITS_PER_WORD - 1) / BITS_PER_WORD, 0) {}
    
    // Accessors with bounds checking
    bool get_bit(size_t idx) const {
      assert(idx < size());
      return (data_[word_index(idx)] >> bit_offset(idx)) & 1;
    }
    
    void set_bit(size_t idx, bool value) {
      assert(idx < size());
      const size_t word = word_index(idx);
      const size_t offset = bit_offset(idx);
      data_[word] = (data_[word] & ~(1ULL << offset)) | (static_cast<uint64_t>(value) << offset);
    }
    
    // Core operations
    void bind(const BinaryHyperVector& other);
    void bundle(const BinaryHyperVector& other);
    void permute(size_t shift);
    
    // Utility methods
    size_t hamming_distance(const BinaryHyperVector& other) const;
    float cosine_similarity(const BinaryHyperVector& other) const;
    size_t size() const { return data_.size() * BITS_PER_WORD; }
  };
  ```

- **Testing Requirements**:
  - Unit tests for bit manipulation correctness
  - Property-based tests for algebraic properties (e.g., binding invertibility)
  - Performance benchmarks for core operations
  - Memory usage validation

- **Code Quality**:
  - Document pre/post-conditions and complexity guarantees
  - Add static assertions for template parameters
  - Ensure exception safety guarantees
  - Include SIMD optimization hints where applicable

### 1.2 Encoders

#### 1.2.1 RandomBasisEncoder
- **Implementation**:
  ```cpp
  template <size_t Dim>
  class RandomBasisEncoder : public Encoder<Dim> {
    std::unordered_map<Symbol, core::HyperVector<Dim, bool>> basis_;
    std::mt19937_64 rng_;
    
  public:
    explicit RandomBasisEncoder(uint64_t seed) : rng_(seed) {}
    
    const core::HyperVector<Dim, bool>& encode(Symbol symbol) override {
      auto it = basis_.find(symbol);
      if (it == basis_.end()) {
        // Generate new random hypervector with good statistical properties
        core::HyperVector<Dim, bool> hv;
        for (size_t i = 0; i < Dim; ++i) {
          hv.set(i, (rng_() % 2) == 1);
        }
        it = basis_.emplace(symbol, std::move(hv)).first;
      }
      return it->second;
    }
    
    // Streaming interface
    void reset() override { /* Reset internal state */ }
    void update(const Symbol& symbol) override { /* Update internal state */ }
    void finalize(core::HyperVector<Dim, bool>* out) override { /* Produce final HV */ }
  };
  ```

- **Testing Requirements**:
  - Deterministic output for same seed
  - Orthogonality testing for different symbols
  - Memory usage validation
  - Thread safety verification

- **Code Quality**:
  - Document statistical properties
  - Add bounds checking in debug builds
  - Include performance counters
  - Add support for serialization/deserialization

## 2. Performance Optimization

### 2.1 SIMD Acceleration

#### 2.1.1 AVX2 Backend
- **Implementation Status**:
  ```cpp
  // Example: AVX2-optimized binding operation
  // Benchmarked at 12.8 GB/s on Intel i9-11900K
  // Memory-bound operation (90% peak bandwidth)
  ```
  
- **Optimization Opportunities**:
  - [ ] Add prefetching for better cache utilization
  - [ ] Explore non-temporal stores for large vectors
  - [ ] Add support for AVX-512 when available

#### 2.1.2 SSE2 Backend
- **Implementation Status**:
  - Basic operations implemented
  - ~60% of AVX2 throughput
  
- **Improvement Areas**:
  - [ ] Optimize memory access patterns
  - [ ] Add support for unaligned loads/stores
  - [ ] Improve instruction-level parallelism

### 2.2 Performance Modeling

#### 2.2.1 Roofline Model
```
Theoretical Peak (AVX2):
- 16 FLOPs/cycle (2x 256-bit FMA)
- 64 B/cycle memory bandwidth
- Expected throughput: 12-14 GB/s on modern CPUs
```

#### 2.2.2 Bottleneck Analysis
1. **Memory-Bound**:
   - Binding: 1 load + 1 store per operation
   - Solution: Cache blocking for large vectors
   
2. **Compute-Bound**:
   - Bundling: Majority voting
   - Solution: Bit-parallel algorithms

### 2.3 GPU Acceleration (Future)
- [ ] CUDA kernel prototypes
- [ ] Memory transfer optimization
- [ ] Multi-GPU support

### 2.1 AVX2 Backend
- **Implementation**:
  ```cpp
  namespace hyperstream::backend::avx2 {
  
  template <size_t Dim>
  void BindAVX2(const core::HyperVector<Dim, bool>& a,
               const core::HyperVector<Dim, bool>& b,
               core::HyperVector<Dim, bool>* out) {
    static_assert(Dim % 256 == 0, "Dimension must be multiple of 256 for AVX2");
    
    const uint64_t* a_data = a.data();
    const uint64_t* b_data = b.data();
    uint64_t* out_data = out->data();
    
    for (size_t i = 0; i < Dim / 64; i += 4) {
      __m256i a_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(a_data + i));
      __m256i b_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(b_data + i));
      __m256i res = _mm256_xor_si256(a_vec, b_vec);
      _mm256_store_si256(reinterpret_cast<__m256i*>(out_data + i), res);
    }
  }
  
  } // namespace hyperstream::backend::avx2
  ```

- **Testing Requirements**:
  - Bit-exact match with scalar implementation
  - Benchmark against scalar implementation
  - Test edge cases (aligned/unaligned memory)
  - Verify correct handling of partial vectors

- **Code Quality**:
  - Platform-specific compilation guards
  - Fallback to scalar implementation
  - Detailed performance documentation
  - Alignment assertions

## 3. Edge & Embedded Support

### 3.1 MCU Optimization

#### 3.1.1 Memory-Efficient Implementation
```c
// Configuration for ARM Cortex-M4 (example)
#define HDC_DIM 2048         // Reduced from 10k for MCUs
#define HDC_WORD_SIZE 32     // Native word size
#define HDC_USE_UNARY_ENCODING 1  // Save power

// Statically allocated storage
typedef struct {
    uint32_t data[HDC_DIM / HDC_WORD_SIZE];
    uint16_t dim;
    uint8_t ref_count;  // For memory management
} hdc_vector_t;
```

#### 3.1.2 Performance Targets
| Operation | Target (Cortex-M4) | Status |
|-----------|-------------------|--------|
| Binding   | 10 μs/op          | ✅ 8 μs |
| Bundling  | 25 μs/op          | ⚠️ 32 μs |
| Similarity| 15 μs/op          | ❌ 45 μs |

### 3.2 Cross-Platform Compatibility
- [ ] Endianness handling
- [ ] Alignment requirements
- [ ] Compiler-specific optimizations

### 3.1 MCU Support
- **Implementation**:
  ```c
  // C-compatible API for embedded systems
  #ifdef __cplusplus
  extern "C" {
  #endif
  
  typedef struct hdc_encoder hdc_encoder_t;
  
  // Initialize encoder with pre-allocated buffer
  hdc_encoder_t* hdc_encoder_init(void* buffer, size_t buffer_size);
  
  // Process input sample
  void hdc_encoder_update(hdc_encoder_t* enc, const uint8_t* sample, size_t sample_size);
  
  // Get resulting hypervector
  const uint8_t* hdc_encoder_finalize(hdc_encoder_t* enc);
  
  #ifdef __cplusplus
  }
  #endif
  ```

- **Testing Requirements**:
  - Memory usage validation
  - Cross-platform testing
  - Performance profiling
  - Power consumption measurement

- **Code Quality**:
  - MISRA-C compliance
  - No dynamic memory allocation
  - Bounds checking
  - Documentation of memory requirements

## 4. Quality Assurance & Testing

### 4.1 Testing Strategy

#### 4.1.1 Unit Testing
```cpp
TEST(HyperVectorTest, BindingInvertibility) {
    BinaryHyperVector<1024> a, b, c;
    // Initialize a, b with random data
    
    // Test: (a ^ b) ^ b == a
    auto bound = a.bind(b).bind(b);
    EXPECT_EQ(a, bound) << "Binding should be self-inverse";
    
    // Performance test
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; ++i) {
        bound = a.bind(b);
    }
    auto duration = /* calculate duration */;
    EXPECT_LT(duration, 1000) << "Binding operation too slow";
}
```

#### 4.1.2 Property-Based Testing
- [ ] Test algebraic properties (associativity, distributivity)
- [ ] Verify statistical properties of random vectors
- [ ] Check edge cases (empty vectors, max dimension)

### 4.2 Code Quality Metrics
| Metric | Target | Current |
|--------|--------|---------|
| Test Coverage | ≥ 90% | 78% |
| Cyclomatic Complexity | ≤ 10 | 8.2 avg |
| Compiler Warnings | 0 | 12 |
| Static Analysis Issues | 0 | 5 |

### 4.3 Fuzz Testing
- [ ] Add libFuzzer targets
- [ ] Test with random inputs
- [ ] Validate memory safety

### 4.1 Testing Strategy
- **Unit Tests**:
  - Google Test framework
  - 100% line coverage for core operations
  - Fuzz testing for input validation
  
- **Integration Tests**:
  - End-to-end pipeline validation
  - Cross-platform consistency checks
  - Performance regression testing

### 4.2 Code Quality
- **Static Analysis**:
  - clang-tidy with custom rules
  - cppcheck for potential bugs
  - Coverity Scan integration
  
- **Documentation**:
  - Doxygen for API documentation
  - Architecture decision records (ADRs)
  - Performance characteristics

## 5. CI/CD & Automation

### 5.1 Build Matrix

#### 5.1.1 Supported Platforms
| Platform | Compiler | Status |
|----------|----------|--------|
| Linux x86_64 | GCC 9-12 | ✅ |
| Linux x86_64 | Clang 10-15 | ✅ |
| Windows x86_64 | MSVC 2019+ | ⚠️ Partial |
| macOS ARM64 | Apple Clang | ❌ Not tested |
| ARM Cortex-M | ARM GCC | ⚠️ Experimental |

#### 5.1.2 Build Configurations
- [x] Debug with sanitizers (ASan, UBSan)
- [x] Release with optimizations
- [ ] MinSizeRel for embedded
- [ ] Coverage builds

### 5.2 Performance Regression Testing
```yaml
# .github/workflows/benchmark.yml
jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run benchmarks
        run: |
          mkdir build && cd build
          cmake -DBUILD_BENCHMARKS=ON ..
          make -j$(nproc)
          ./benchmarks/benchmark_hamming
          # Compare against baseline
          python3 ../scripts/check_perf_regression.py results.json baseline.json
```

### 5.1 Build Matrix
- **Platforms**:
  - Linux (GCC/Clang)
  - Windows (MSVC)
  - macOS (Clang)
  - ARM Cortex-M (cross-compilation)

- **Build Types**:
  - Debug with sanitizers
  - Release with optimizations
  - Coverage builds

### 5.2 Deployment
- **Artifacts**:
  - Static/dynamic libraries
  - Python wheels
  - Docker images
  - Documentation site

## 6. Advanced Optimization

### 6.1 Memory Access Patterns

#### 6.1.1 Cache Optimization
```cpp
// Process data in cache-friendly blocks
constexpr size_t CACHE_LINE_SIZE = 64; // bytes
constexpr size_t ELEMENTS_PER_CACHE_LINE = CACHE_LINE_SIZE / sizeof(uint64_t);

for (size_t i = 0; i < size; i += ELEMENTS_PER_CACHE_LINE) {
    // Prefetch next cache line
    if (i + ELEMENTS_PER_CACHE_LINE < size) {
        __builtin_prefetch(&data[i + ELEMENTS_PER_CACHE_LINE]);
    }
    
    // Process current cache line
    for (size_t j = 0; j < ELEMENTS_PER_CACHE_LINE && (i + j) < size; ++j) {
        // Process element
    }
}
```

#### 6.1.2 SIMD Optimization
- [ ] Align memory to cache lines
- [ ] Use gather/scatter for sparse operations
- [ ] Exploit SIMD for bit manipulation

### 6.2 Parallelism

#### 6.2.1 Multi-threading
```cpp
// Thread pool for parallel operations
class ThreadPool {
    // Implementation details...
    
    template<typename Func, typename... Args>
    auto enqueue(Func&& f, Args&&... args) {
        // Queue task and return future
    }
};

// Usage:
ThreadPool pool(std::thread::hardware_concurrency());
std::vector<std::future<void>> results;

for (auto& vec : vectors) {
    results.emplace_back(pool.enqueue([&] {
        process_vector(vec);
    }));
}

// Wait for completion
for (auto& result : results) {
    result.get();
}
```

#### 6.2.2 Vectorization
- [ ] Auto-vectorization hints
- [ ] Explicit SIMD intrinsics
- [ ] Compiler-specific optimizations

### 6.1 Profiling
- **Tools**:
  - Google Benchmark
  - Intel VTune
  - perf (Linux)
  - Xcode Instruments (macOS)

### 6.2 Optimization Targets
- **Memory Access Patterns**:
  - Cache line alignment
  - Prefetching
  - Structure of Arrays (SoA) layout

- **Parallelism**:
  - Thread pool implementation
  - Task-based parallelism
  - SIMD vectorization

## 7. Documentation & Knowledge Base

### 7.1 API Documentation

#### 7.1.1 Doxygen Style
```cpp
/**
 * @brief Binds two hypervectors using XOR operation
 * 
 * @tparam Dim Dimension of the hypervectors
 * @param a First input hypervector
 * @param b Second input hypervector
 * @param[out] out Result of binding (a ^ b)
 * 
 * @note This operation is its own inverse: bind(a, b, tmp); bind(tmp, b) == a
 * @throws std::invalid_argument If dimensions don't match
 * 
 * @complexity O(Dim / 64) for binary hypervectors
 * @memory Accesses 3 * (Dim / 8) bytes
 */
template<size_t Dim>
void bind(const HyperVector<Dim, bool>& a, 
          const HyperVector<Dim, bool>& b,
          HyperVector<Dim, bool>* out);
```

### 7.2 Performance Documentation
- [ ] Operation timing tables
- [ ] Memory usage guidelines
- [ ] Platform-specific notes

### 7.3 Tutorials
- [ ] Getting started guide
- [ ] Performance tuning cookbook
- [ ] Edge deployment walkthrough

### 7.1 Developer Documentation
- **Code Organization**:
  - Namespace hierarchy
  - Module dependencies
  - Build system overview

- **API Reference**:
  - Class diagrams
  - Method documentation
  - Example usage

### 7.2 User Guide
- **Getting Started**:
  - Installation instructions
  - Basic usage examples
  - Troubleshooting guide

- **Advanced Topics**:
  - Custom encoder implementation
  - Performance tuning
  - Platform-specific optimizations

## 8. Roadmap & Future Work

### 8.1 Short-term (Next 3 Months)

#### 8.1.1 Core Functionality
- [x] Basic hypervector operations
- [ ] Complete encoder implementations
- [ ] Optimize associative memory

#### 8.1.2 Performance
- [ ] Achieve 90% of theoretical peak
- [ ] Add ARM NEON backend
- [ ] Optimize for embedded targets

### 8.2 Medium-term (3-6 Months)
- [ ] GPU acceleration
- [ ] Advanced encoders
- [ ] Streaming API

### 8.3 Long-term (6+ Months)
- [ ] Neuromorphic backends
- [ ] Distributed training
- [ ] Quantum computing integration

## 9. Risk Management

### 9.1 Identified Risks
| Risk | Impact | Mitigation | Status |
|------|--------|------------|--------|
| Performance regressions | High | Automated benchmarking | ✅ In place |
| Memory leaks | Critical | Sanitizers, RAII | ⚠️ Needs review |
| Platform compatibility | Medium | CI/CD matrix | ⚠️ In progress |
| API stability | High | Versioning, deprecation policy | ❌ Needed |

### 9.2 Performance Budget
| Operation | Target Latency | Current | Notes |
|-----------|----------------|---------|-------|
| Bind (10k dim) | 500 ns | 420 ns | ✅ |
| Bundle (10 vectors) | 5 μs | 7.2 μs | ⚠️ Needs work |
| Similarity (1M pairs/s) | 1 μs/pair | 1.8 μs/pair | ❌ Critical |
| AM Classify (10k dim, 256 entries) | ≥40k qps | 46.2k qps | ✅ PrototypeMemory core; SSE2/AVX2 A/B pending |

### 8.1 Short-term
- [ ] Implement remaining core operations
- [ ] Add more encoder implementations
- [ ] Expand test coverage

### 8.2 Medium-term
- [ ] GPU acceleration
- [ ] Neuromorphic backends
- [ ] Advanced encoders

### 8.3 Long-term
- [ ] Quantum computing integration
- [ ] Photonic accelerators
- [ ] Distributed training

## 9. Changelog

### [Unreleased]
- Initial implementation plan
- Core architecture documentation
- Testing infrastructure setup

## 10. Contributing

### 10.1 Development Workflow
1. Create feature branch from `main`
2. Write tests for new functionality
3. Implement changes
4. Run code formatters and linters
5. Submit pull request

### 10.2 Code Review
- Required approvals: 2
- Required checks:
  - All tests passing
  - Code coverage >= 90%
  - Static analysis clean
  - Documentation updated

## 11. License

[Specify License]

## 12. References

1. Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors
2. A Theoretical Perspective on Hyperdimensional Computing
3. Hardware Optimizations of Dense Binary Hyperdimensional Computing: From CMOS to Compute-in-Memory to Neuro-Inspired Architectures

<!-- Link reference definitions to satisfy linters -->

[Unreleased]: https://github.com/your-org/hyperstream/compare/v0.0.0...HEAD
[Specify License]: #11-license

## Progress Update (2025-10-13)

- Batch 3 cleanup complete: clang-tidy findings addressed per-file; C++17 nested namespaces; module headers added; serialization/memory APIs clarified.
- CI stable across OS matrix with perf/golden gates; NEON integrated.
- Benchmarks and determinism scaffolding complete; snapshot/restore parity landed.

## Next Steps

1. .clang-tidy configuration
   - Convert `Checks` to YAML list form; consider enabling `cppcoreguidelines-*`, `cert-*`; keep stylistic exceptions as needed.
2. Performance tuning
   - Revisit AVX2 bind thresholds; document size-based heuristics; A/B small-dim `loadu`-only vs aligned loads.
   - Add Hamming SSE2/AVX2 auto-tune guidance; expose/env override already present.
3. Associative memory ergonomics
   - Decide inline vs heap for large `Dim*Capacity` (ADR + tests); ensure no stack-pressure construction sites.
4. API documentation
   - Ensure ADR-0001 is referenced in README/Docs where applicable; add examples for new args structs (`HashEncoderConfig`, `TokenRole`, `*Args`).
4. Coverage expansion
   - Unit tests for IO failure paths; more property tests (bundling invariants); fuzz HSER1 corruption.
5. Documentation
   - Expand standardized headers to any remaining files; add ADR for policy threshold decision; run doc linters.
