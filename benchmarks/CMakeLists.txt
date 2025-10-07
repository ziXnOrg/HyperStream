# Benchmarks CMake

add_executable(bind_bench
  bind_bench.cpp
)

# Header-only interface target supplies include dirs
target_link_libraries(bind_bench PRIVATE hyperstream)

# Strict warnings for benchmarks as well
if(MSVC)
  target_compile_options(bind_bench PRIVATE /W4 /WX)
else()
  target_compile_options(bind_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(am_bench
  am_bench.cpp
)

target_link_libraries(am_bench PRIVATE hyperstream)

if(MSVC)
  target_compile_options(am_bench PRIVATE /W4 /WX)
else()
  target_compile_options(am_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(cluster_bench
  cluster_bench.cpp
)

target_link_libraries(cluster_bench PRIVATE hyperstream)

if(MSVC)
  target_compile_options(cluster_bench PRIVATE /W4 /WX)
else()
  target_compile_options(cluster_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(permute_bench
  permute_bench.cpp
)

target_link_libraries(permute_bench PRIVATE hyperstream)

if(MSVC)
  target_compile_options(permute_bench PRIVATE /W4 /WX)
else()
  target_compile_options(permute_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

add_executable(hamming_bench
  hamming_bench.cpp
)

target_link_libraries(hamming_bench PRIVATE hyperstream)

if(MSVC)
  target_compile_options(hamming_bench PRIVATE /W4 /WX)
else()
  target_compile_options(hamming_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()

# Config report benchmark
add_executable(config_bench
  config_bench.cpp
)

target_link_libraries(config_bench PRIVATE hyperstream)

if(MSVC)
  target_compile_options(config_bench PRIVATE /W4 /WX)
else()
  target_compile_options(config_bench PRIVATE -Wall -Wextra -Wpedantic -Werror)
endif()
