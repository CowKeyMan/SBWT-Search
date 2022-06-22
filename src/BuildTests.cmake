# Builds the testing program. We use googletest as a testing framework

if (BUILD_TESTS)

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(test_lib INTERFACE)
target_link_libraries(
  test_lib INTERFACE
  gtest
  gtest_main
  gmock
  gmock_main
  gcov
)

set(
  test_common_sources
  "${PROJECT_SOURCE_DIR}/test_main.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/BenchmarkUtils_test.cpp"
  "${PROJECT_SOURCE_DIR}/QueryFileParser/QueryFileParser_test.cpp"
  "${PROJECT_SOURCE_DIR}/RawSequencesParser/RawSequencesParser_test.cpp"
)
set(
  test_common_include_dirs
  "${PROJECT_SOURCE_DIR}/Utils/"
  "${PROJECT_SOURCE_DIR}/Global/"
)


# Create cpu test executable
if (BUILD_CPU)
  add_executable(
    test_main_cpu
    ${test_common_sources}
  )
  target_include_directories(
    test_main_cpu
    PRIVATE
    ${test_common_include_dirs}
  )
  add_test(NAME test_main_cpu COMMAND test_main_cpu)
  target_link_libraries(
    test_main_cpu
    PRIVATE common_options
    PRIVATE libraries_cpu
    PRIVATE test_lib
  )

endif()

if (CMAKE_CUDA_COMPILER AND BUILD_CUDA)

# Create cuda test executable
  add_executable(
    test_main_cuda
    ${test_common_sources}
    # "${other test sources}"
  )
  target_include_directories(
    test_main_cuda
    PRIVATE
    ${test_common_include_dirs}
  )
  add_test(NAME all_tests COMMAND test_main_cuda)
  target_link_libraries(
    test_main_cuda
    PRIVATE common_options
    PRIVATE libraries_cuda
    PRIVATE test_lib
  )
endif()

endif()
