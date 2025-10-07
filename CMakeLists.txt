cmake_minimum_required(VERSION 3.20)
project(HyperStream VERSION 0.1.0 LANGUAGES CXX)

# Enforce C++ standard and compile commands for tooling
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(HYPERSTREAM_ENABLE_TESTS "Build tests" ON)
option(HYPERSTREAM_ENABLE_BENCHMARKS "Build benchmarks" ON)

# Compiler warnings: strict, fail on warnings
if(MSVC)
  add_compile_options(/W4 /WX)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else()
  add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Enable SIMD optimizations when supported
include(CheckCXXCompilerFlag)
if(MSVC)
  check_cxx_compiler_flag(/arch:AVX2 COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(/arch:AVX2)
  endif()
else()
  check_cxx_compiler_flag(-mavx2 COMPILER_SUPPORTS_AVX2)
  if(COMPILER_SUPPORTS_AVX2)
    add_compile_options(-mavx2)
  endif()
endif()

# Header-only interface target for HyperStream core
add_library(hyperstream INTERFACE)

target_include_directories(hyperstream INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

# TODO: When sources exist, add compiled library targets for backends.

# Tests
if(HYPERSTREAM_ENABLE_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# Benchmarks
if(HYPERSTREAM_ENABLE_BENCHMARKS)
  add_subdirectory(benchmarks)
endif()


# Note: To enable wide bundler counters (32-bit, no saturation) for niche scenarios,
# pass a compiler define for all targets, for example:
#   - GCC/Clang: cmake -S . -B build -DCMAKE_CXX_FLAGS="-DHYPERSTREAM_BUNDLER_COUNTER_WIDE"
#   - MSVC (Developer Command Prompt): cmake -S . -B build -DCMAKE_CXX_FLAGS="/DHYPERSTREAM_BUNDLER_COUNTER_WIDE"
# Default build uses 16-bit saturating counters.
