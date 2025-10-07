# Tests CMake

# Require vendored GoogleTest in third_party/googletest to avoid network fetch during build.
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/third_party/googletest/CMakeLists.txt")
  message(FATAL_ERROR "third_party/googletest not found. Fetch it locally before building tests.")
endif()
add_subdirectory(${CMAKE_SOURCE_DIR}/third_party/googletest ${CMAKE_BINARY_DIR}/third_party/googletest)

add_executable(core_ops_tests
  core_ops_tests.cc
)

target_link_libraries(core_ops_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

# Strict warnings for tests as well
if(MSVC)
  target_compile_options(core_ops_tests PRIVATE /W4 /WX)
else()
  target_compile_options(core_ops_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

include(GoogleTest)
gtest_discover_tests(core_ops_tests)

add_executable(dispatch_tests
  dispatch_tests.cc
)

target_link_libraries(dispatch_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(dispatch_tests PRIVATE /W4 /WX)
else()
  target_compile_options(dispatch_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(dispatch_tests)

add_executable(forced_scalar_tests
  forced_scalar_tests.cc
)

target_compile_definitions(forced_scalar_tests PRIVATE HYPERSTREAM_FORCE_SCALAR=1)

target_link_libraries(forced_scalar_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(forced_scalar_tests PRIVATE /W4 /WX)
else()
  target_compile_options(forced_scalar_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(forced_scalar_tests)

add_executable(encoding_tests
  encoding_tests.cc
)

target_link_libraries(encoding_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(encoding_tests PRIVATE /W4 /WX)
else()
  target_compile_options(encoding_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(encoding_tests)

add_executable(associative_tests
  associative_tests.cc
)

target_link_libraries(associative_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(associative_tests PRIVATE /W4 /WX)
else()
  target_compile_options(associative_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(associative_tests)

add_executable(backend_tests
  backend_tests.cc
)

target_link_libraries(backend_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(backend_tests PRIVATE /W4 /WX)
else()
  target_compile_options(backend_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(backend_tests)

# Edge case tests
add_executable(edge_case_tests
  edge_case_tests.cc
)

target_link_libraries(edge_case_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(edge_case_tests PRIVATE /W4 /WX)
else()
  target_compile_options(edge_case_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(edge_case_tests)

# Wide-counter smoke test (compiles with HYPERSTREAM_BUNDLER_COUNTER_WIDE)
add_executable(wide_counter_smoke_tests
  wide_counter_smoke_tests.cc
)

target_link_libraries(wide_counter_smoke_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

# Compile with macro enabled for this target only
if(MSVC)
  target_compile_definitions(wide_counter_smoke_tests PRIVATE HYPERSTREAM_BUNDLER_COUNTER_WIDE)
  target_compile_options(wide_counter_smoke_tests PRIVATE /W4 /WX)
else()
  target_compile_definitions(wide_counter_smoke_tests PRIVATE HYPERSTREAM_BUNDLER_COUNTER_WIDE)
  target_compile_options(wide_counter_smoke_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(wide_counter_smoke_tests)



# Config tests
add_executable(config_tests
  config_tests.cc
)

target_link_libraries(config_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(config_tests PRIVATE /W4 /WX)
else()
  target_compile_options(config_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(config_tests)

# Capability tests
add_executable(capability_tests
  capability_tests.cc
)

target_link_libraries(capability_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(capability_tests PRIVATE /W4 /WX)
else()
  target_compile_options(capability_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(capability_tests)

# Policy tests
add_executable(policy_tests
  policy_tests.cc
)

target_link_libraries(policy_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

if(MSVC)
  target_compile_options(policy_tests PRIVATE /W4 /WX)
else()
  target_compile_options(policy_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

gtest_discover_tests(policy_tests)


# Property-based dispatch invariants
add_executable(dispatch_property_tests
  dispatch_property_tests.cc
)

if(MSVC)
  target_compile_options(dispatch_property_tests PRIVATE /W4 /WX)
else()
  target_compile_options(dispatch_property_tests PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

target_link_libraries(dispatch_property_tests PRIVATE
  hyperstream
  gtest
  gtest_main
)

gtest_discover_tests(dispatch_property_tests)
