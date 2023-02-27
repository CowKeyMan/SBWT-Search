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


set(
  gpu_test_sources
  "${PROJECT_SOURCE_DIR}/UtilityKernels/Rank_test.cu"
  "${PROJECT_SOURCE_DIR}/UtilityKernels/GetBoolFromBitVector_test.cu"
  "${PROJECT_SOURCE_DIR}/UtilityKernels/VariableLengthIntegerIndex_test.cu"
)
add_library(
  gpu_tests
  ${gpu_test_sources}
)
target_link_libraries(gpu_tests PRIVATE gpu_utils)
set_source_files_properties(
  ${gpu_test_sources}
  TARGET_DIRECTORY gpu_tests
  PROPERTIES LANGUAGE ${HIP_TARGET_LANGUAGE}
)

# Create cpu test executable
add_executable(
  test_main
  "${PROJECT_SOURCE_DIR}/test_main.cpp"

  "${PROJECT_SOURCE_DIR}/Tools/CircularQueue_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/CircularBuffer_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/IOUtils_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/Semaphore_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/MathUtils_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/Logger_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/MemoryUnitsParser_test.cpp"
  "${PROJECT_SOURCE_DIR}/Tools/BenchmarkUtils_test.cpp"

  "${PROJECT_SOURCE_DIR}/FilenamesParser/FilenamesParser_test.cpp"

  "${PROJECT_SOURCE_DIR}/SequenceFileParser/ContinuousSequenceFileParser_test.cpp"
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/PositionsBuilder_test.cpp"
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/ContinuousPositionsBuilder_test.cpp"
  "${PROJECT_SOURCE_DIR}/SeqToBitsConverter/ContinuousSeqToBitsConverter_test.cpp"

  "${PROJECT_SOURCE_DIR}/ColorIndexBuilder/ColorIndexBuilder_test.cpp"
  "${PROJECT_SOURCE_DIR}/IndexFileParser/IndexFileParserTestUtils.cpp"
  "${PROJECT_SOURCE_DIR}/IndexFileParser/AsciiIndexFileParser_test.cpp"
  "${PROJECT_SOURCE_DIR}/IndexFileParser/BinaryIndexFileParser_test.cpp"
  "${PROJECT_SOURCE_DIR}/IndexFileParser/ContinuousIndexFileParser_test.cpp"

  "${PROJECT_SOURCE_DIR}/UtilityKernels/Rank_test.cpp"
  "${PROJECT_SOURCE_DIR}/UtilityKernels/GetBoolFromBitVector_test.cpp"
  "${PROJECT_SOURCE_DIR}/UtilityKernels/VariableLengthIntegerIndex_test.cpp"
)
add_test(NAME test_main COMMAND test_main)
target_link_libraries(
  test_main
  PRIVATE
  common_libraries
  test_lib
  gpu_tests
)

endif() # BUILD_TESTS
