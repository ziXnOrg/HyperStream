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

# SIMD codegen policy
# - GCC/Clang: use function-level target attributes in headers for ISA-specific codepaths
# - MSVC: build per-ISA object libraries from dedicated translation units with per-file /arch flags

# Header-only interface target for HyperStream core
add_library(hyperstream INTERFACE)

# MSVC-only (x86/x64): add per-ISA object libraries and aggregate static lib for linking
if(MSVC AND (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|i686|i386"))
  file(GLOB HS_SSE2_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/*sse2.cpp)
  file(GLOB HS_AVX2_SRCS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/backend/*avx2.cpp)

  add_library(hs_kernels_sse2 OBJECT ${HS_SSE2_SRCS})
  target_compile_options(hs_kernels_sse2 PRIVATE /arch:SSE2)
  target_include_directories(hs_kernels_sse2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  add_library(hs_kernels_avx2 OBJECT ${HS_AVX2_SRCS})
  # Guard: only add /arch:AVX2 if supported
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag(/arch:AVX2 HS_MSVC_HAS_AVX2)
  if(HS_MSVC_HAS_AVX2)
    target_compile_options(hs_kernels_avx2 PRIVATE /arch:AVX2)
  else()
    message(WARNING "Compiler does not support /arch:AVX2; AVX2 TUs may fail to build.")
  endif()
  target_include_directories(hs_kernels_avx2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

  add_library(hyperstream_kernels STATIC
    $<TARGET_OBJECTS:hs_kernels_sse2>
    $<TARGET_OBJECTS:hs_kernels_avx2>
  )
  # Propagate kernels to consumers of the header-only interface on MSVC
  target_link_libraries(hyperstream INTERFACE hyperstream_kernels)
endif()


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
