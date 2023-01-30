# Builds items which are commonly used between the main program and the tests.
# Usually these are classes, files and options which are used by the main program but
# are also tested individually.
# Any common options are put as an interface
# rather than putting it with each file individually

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# External Dependencies
include(ExternalProject)
include(FetchContent)

## Fetch kseqpp_read
FetchContent_Declare(
  reklibpp
  QUIET
  GIT_REPOSITORY       "https://github.com/CowKeyMan/kseqpp_REad"
  GIT_TAG              v1.2.0
  GIT_SHALLOW          TRUE
)
FetchContent_MakeAvailable(reklibpp)

## Fetch cxxopts
FetchContent_Declare(
  cxxopts
  GIT_REPOSITORY       https://github.com/jarro2783/cxxopts
  GIT_TAG              v3.0.0
  GIT_SHALLOW          TRUE
)
set(CXXOPTS_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CXXOPTS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(CXXOPTS_ENABLE_WARNINGS OFF CACHE BOOL "" FORCE)
include_directories("${CMAKE_BINARY_DIR}/deps/cxxopts-src/include")
FetchContent_MakeAvailable(cxxopts)

# Fetch OpenMP
find_package(OpenMP REQUIRED)
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")

add_library(
  argument_parser
  "${PROJECT_SOURCE_DIR}/ArgumentParser/ArgumentParser.cpp"
)
target_link_libraries(argument_parser PRIVATE cxxopts memory_units_parser)
add_library(
  presearcher_cpu
  "${PROJECT_SOURCE_DIR}/Presearcher/Presearcher.cpp"
)
target_link_libraries(presearcher_cpu PRIVATE cuda_utils)
add_library(
  presearcher_cuda
  "${PROJECT_SOURCE_DIR}/Presearcher/Presearcher.cu"
)
set_target_properties(presearcher_cuda PROPERTIES CUDA_ARCHITECTURES "80;70;60")
target_link_libraries(presearcher_cuda PRIVATE cuda_utils)
add_library(
  sequence_file_parser
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/ContinuousSequenceFileParser.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/IntervalBatchProducer.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/StringBreakBatchProducer.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/StringSequenceBatchProducer.cpp"
)
target_link_libraries(
  sequence_file_parser
  PRIVATE
  kseqpp_read
  io_utils
  error_utils
  fmt::fmt
  logger
  OpenMP::OpenMP_CXX
)
add_library(
  seq_to_bits_converter
  "${PROJECT_SOURCE_DIR}/SeqToBitsConverter/ContinuousSeqToBitsConverter.cpp"
  "${PROJECT_SOURCE_DIR}/SeqToBitsConverter/InvalidCharsProducer.cpp"
  "${PROJECT_SOURCE_DIR}/SeqToBitsConverter/BitsProducer.cpp"
)
target_link_libraries(
  seq_to_bits_converter PRIVATE fmt::fmt logger OpenMP::OpenMP_CXX
)
add_library(
  filenames_parser
  "${PROJECT_SOURCE_DIR}/FilenamesParser/FilenamesParser.cpp"
)
add_library(
  poppy_builder
  "${PROJECT_SOURCE_DIR}/PoppyBuilder/PoppyBuilder.cpp"
)
add_library(
  sbwt_builder
  "${PROJECT_SOURCE_DIR}/SbwtBuilder/SbwtBuilder.cpp"
)
target_link_libraries(
  sbwt_builder
  PRIVATE
  io_utils
  OpenMP::OpenMP_CXX
  fmt::fmt
  cuda_utils
)
add_library(
  sbwt_container
  "${PROJECT_SOURCE_DIR}/SbwtContainer/SbwtContainer.cpp"
  "${PROJECT_SOURCE_DIR}/SbwtContainer/CpuSbwtContainer.cpp"
  "${PROJECT_SOURCE_DIR}/SbwtContainer/GpuSbwtContainer.cpp"
)
target_link_libraries(sbwt_container PUBLIC cuda_utils)
add_library(
  positions_builder
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/PositionsBuilder.cpp"
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/ContinuousPositionsBuilder.cpp"
)
target_link_libraries(
  positions_builder PRIVATE fmt::fmt logger OpenMP::OpenMP_CXX
)
add_library(
  output_parser
  "${PROJECT_SOURCE_DIR}/OutputParser/AsciiOutputParser.cpp"
  "${PROJECT_SOURCE_DIR}/OutputParser/BinaryOutputParser.cpp"
  "${PROJECT_SOURCE_DIR}/OutputParser/BoolOutputParser.cpp"
)
target_link_libraries(output_parser PRIVATE io_utils)
add_library(
  searcher_cpu
  "${PROJECT_SOURCE_DIR}/Searcher/Searcher.cpp"
)
target_link_libraries(searcher_cpu PRIVATE fmt::fmt)
add_library(
  searcher_cuda
  "${PROJECT_SOURCE_DIR}/Searcher/Searcher.cu"
)
set_target_properties(searcher_cuda PROPERTIES CUDA_ARCHITECTURES "80;70;60")
target_link_libraries(searcher_cuda PRIVATE fmt::fmt cuda_utils)
add_library(
  continuous_searcher
  "${PROJECT_SOURCE_DIR}/Searcher/ContinuousSearcher.cpp"
)
target_link_libraries(continuous_searcher PRIVATE fmt::fmt searcher_cpu searcher_cuda)
add_library(
  results_printer
  "${PROJECT_SOURCE_DIR}/ResultsPrinter/AsciiContinuousResultsPrinter.cpp"
  "${PROJECT_SOURCE_DIR}/ResultsPrinter/BinaryContinuousResultsPrinter.cpp"
  "${PROJECT_SOURCE_DIR}/ResultsPrinter/BoolContinuousResultsPrinter.cpp"
)
target_link_libraries(results_printer PRIVATE io_utils fmt::fmt)

# Common libraries
add_library(common_libraries INTERFACE)
target_link_libraries(
  common_libraries
  INTERFACE
  # external libraries
  fmt::fmt
  kseqpp_read
  OpenMP::OpenMP_CXX
  cxxopts
  spdlog::spdlog

  # Internal libraries
  io_utils
  logger
  error_utils
  memory_units_parser
  memory_utils
  omp_lock
  semaphore

  ## SBWT_SEARCH libraries
  argument_parser
  filenames_parser
  sbwt_builder
  sbwt_container
  poppy_builder
  results_printer
  presearcher_cpu
  presearcher_cuda

  sequence_file_parser
  seq_to_bits_converter
  positions_builder
  continuous_searcher
)


# Link cuda items
if (CMAKE_CUDA_COMPILER)
  target_link_libraries(common_libraries INTERFACE cuda_utils)
endif()
