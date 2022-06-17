option(
  BUILD_BENCHMARKS
  "Build the benchmark targets"
  ON
)

if(BUILD_BENCHMARKS)

add_executable(benchmark_main benchmark_main.cpp)
target_link_libraries(
  benchmark_main
  PRIVATE libraries_cpu
  PRIVATE libraries_cuda
)
target_include_directories(
  benchmark_main
  PUBLIC "${PROJECT_SOURCE_DIR}/Utils"
)
enable_warnings(benchmark_main "PRIVATE")

endif()
