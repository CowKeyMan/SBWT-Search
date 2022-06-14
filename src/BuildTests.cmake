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

# TODO: Add more sources here
# Create cpu test executable
if (BUILD_CPU)
  add_executable(
    test_main_cpu
    "${PROJECT_SOURCE_DIR}/test_main.cpp"
    # "${other test sources}"
  )
  add_test(NAME test_main_cpu COMMAND test_main_cpu)
  target_link_libraries(
    test_main_cpu
    PRIVATE common_options
    PRIVATE test_lib
    PRIVATE libraries_cpu
  )
endif()

if (CMAKE_CUDA_COMPILER AND BUILD_CUDA)
# Create cuda test executable
  add_executable(
    test_main_cuda
    "${PROJECT_SOURCE_DIR}/test_main.cpp"
    # "${other test sources}"
  )
  add_test(NAME all_tests COMMAND test_main_cuda)
  target_link_libraries(
    test_main_cuda
    PRIVATE common_options
    PRIVATE test_lib
    PRIVATE libraries_cuda
  )
endif()

endif()
