# Builds the testing program. We use googletest as a testing framework

if (BUILD_TESTS)

include(FetchContent)
FetchContent_Declare(
  googletest
  QUIET
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
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

# Create cpu test executable
add_executable(
  test_main
  "${PROJECT_SOURCE_DIR}/test_main.cpp"

  "${PROJECT_SOURCE_DIR}/Utils/CircularQueue_test.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/CircularBuffer_test.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/IOUtils_test.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/Semaphore_test.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/MathUtils_test.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/Logger_test.cpp"
  "${PROJECT_SOURCE_DIR}/Utils/MemoryUnitsParser_test.cpp"

  "${PROJECT_SOURCE_DIR}/SequenceFileParser/ContinuousSequenceFileParser_test.cpp"
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/PositionsBuilder_test.cpp"
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/ContinuousPositionsBuilder_test.cpp"
  "${PROJECT_SOURCE_DIR}/SeqToBitsConverter/ContinuousSeqToBitsConverter_test.cpp"
  "${PROJECT_SOURCE_DIR}/ResultsPrinter/ContinuousResultsPrinter_test.cpp"

  "${PROJECT_SOURCE_DIR}/Utils/BenchmarkUtils_test.cpp"
  "${PROJECT_SOURCE_DIR}/FilenamesParser/FilenamesParser_test.cpp"
)
add_test(NAME test_main COMMAND test_main)
target_link_libraries(
  test_main
  PRIVATE
  common_libraries
  test_lib
)

endif() # BUILD_TESTS
