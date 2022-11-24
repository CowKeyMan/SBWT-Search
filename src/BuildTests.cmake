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
if (BUILD_CPU)
  add_executable(
    test_main_cpu
    "${PROJECT_SOURCE_DIR}/test_main.cpp"

    "${PROJECT_SOURCE_DIR}/Utils/CircularQueue_test.cpp"
    "${PROJECT_SOURCE_DIR}/Utils/CircularBuffer_test.cpp"
    "${PROJECT_SOURCE_DIR}/Utils/IOUtils_test.cpp"
    "${PROJECT_SOURCE_DIR}/Utils/Semaphore_test.cpp"
    "${PROJECT_SOURCE_DIR}/Utils/MathUtils_test.cpp"
    # "${PROJECT_SOURCE_DIR}/Utils/BoundedSemaphore_test.cpp" # BROKEN
    # "${PROJECT_SOURCE_DIR}/Utils/Logger_test.cpp" # BROKEN

    "${PROJECT_SOURCE_DIR}/SequenceFileParser/ContinuousSequenceFileParser_test.cpp"
    "${PROJECT_SOURCE_DIR}/PositionsBuilder/PositionsBuilder_test.cpp"
    "${PROJECT_SOURCE_DIR}/PositionsBuilder/ContinuousPositionsBuilder_test.cpp"
    "${PROJECT_SOURCE_DIR}/SeqToBitsConverter/ContinuousSeqToBitsConverter_test.cpp"

    "${PROJECT_SOURCE_DIR}/ResultsPrinter/ContinuousResultsPrinter_test.cpp"

    # "${PROJECT_SOURCE_DIR}/Utils/BenchmarkUtils_test.cpp"
    # "${PROJECT_SOURCE_DIR}/PoppyBuilder/PoppyBuilder_test.cpp"
    # "${PROJECT_SOURCE_DIR}/TestUtils/RankTestUtils_test.cpp"
    # "${PROJECT_SOURCE_DIR}/FilenamesParser/FilenamesParser_test.cpp"
    # TODO: add more cpu tests here
  )
  add_test(NAME test_main_cpu COMMAND test_main_cpu)
  target_link_libraries(
    test_main_cpu
    PRIVATE
    libraries_cpu
    kseqpp_read
    sequence_file_parser
    test_lib
    OpenMP::OpenMP_CXX
    spdlog::spdlog
  )
endif()

# Create cuda test executable
# if (CMAKE_CUDA_COMPILER AND BUILD_CUDA)
#   add_library(
#     rank_test
#     "${PROJECT_SOURCE_DIR}/Rank/Rank_test.cu"
#   )
#   target_link_libraries(rank_test PRIVATE spdlog::spdlog)
#   set_target_properties(rank_test PROPERTIES CUDA_ARCHITECTURES "80;70;60")
#   add_executable(
#     test_main_cuda
#     "${PROJECT_SOURCE_DIR}/test_main.cpp"
#     "${PROJECT_SOURCE_DIR}/Rank/Rank_test.cpp"
#     # TODO: add more cuda tests here
#   )
#   add_test(NAME all_tests COMMAND test_main_cuda)
#   target_link_libraries(
#     test_main_cuda
#     PRIVATE
#     libraries_cuda
#     test_lib
#     rank_test
#   )
# endif()

endif() # BUILD_TESTS
