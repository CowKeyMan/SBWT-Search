# Builds items which are commonly used between the main program and the tests.
# Usually these are classes, files and options which are used by the main program but
# are also tested individually

# Any common options are put as an interface
# rather than putting it with each file individually

include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")

# common options
add_compile_options(
  "$<$<CONFIG:Debug>:--coverage>" # only in debug mode
)
add_compile_options(
  "$<$<CONFIG:Release>:-O3>" # only in release mode
)
link_libraries(gcov)

# External Dependencies
include(ExternalProject)
include(FetchContent)

FetchContent_Declare(
  fmt
  QUIET
  GIT_REPOSITORY       https://github.com/fmtlib/fmt.git
  GIT_TAG              9.1.0
  GIT_SHALLOW          TRUE
)
FetchContent_MakeAvailable(fmt)

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

# Fetch spdlog
FetchContent_Declare(
  spdlog
  QUIET
  GIT_REPOSITORY       https://github.com/gabime/spdlog
  GIT_TAG              v1.10.0
  GIT_SHALLOW          TRUE
  )
FetchContent_MakeAvailable(spdlog)
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include")

# My libraries
add_library(
  io_utils
  "${PROJECT_SOURCE_DIR}/Utils/IOUtils.cpp"
)
target_link_libraries(
  io_utils
  PRIVATE
  fmt::fmt
)
add_library(
  error_utils
  "${PROJECT_SOURCE_DIR}/Utils/ErrorUtils.cpp"
)
add_library(
  logger
  "${PROJECT_SOURCE_DIR}/Utils/Logger.cpp"
)
target_link_libraries(logger PRIVATE spdlog::spdlog)

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
target_link_libraries(sbwt_builder PRIVATE io_utils OpenMP::OpenMP_CXX)
add_library(
  sbwt_container_cpu
  "${PROJECT_SOURCE_DIR}/SbwtContainer/SbwtContainer.cpp"
)
add_library(
  positions_builder
  "${PROJECT_SOURCE_DIR}/PositionsBuilder/PositionsBuilder.cpp"
)
target_link_libraries(positions_builder PRIVATE OpenMP::OpenMP_CXX)
add_library(
  sbwt_container_gpu
  "${PROJECT_SOURCE_DIR}/SbwtContainer/GpuSbwtContainer.cu"
  "${PROJECT_SOURCE_DIR}/SbwtContainer/CpuSbwtContainer.cu"
)
set_target_properties(sbwt_container_gpu PROPERTIES CUDA_ARCHITECTURES "80;70;60")
add_library(sbwt_container INTERFACE)
target_link_libraries(
  sbwt_container
  INTERFACE
  sbwt_container_cpu
  sbwt_container_gpu
  kseqpp_read
)

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

  # Internal libraries
  io_utils
  logger
  error_utils
  filenames_parser
  sequence_file_parser
  positions_builder

  ## SBWT Loading libraries
  sbwt_builder
  sbwt_container
  poppy_builder
)
add_library(
  cuda_utils
  "${PROJECT_SOURCE_DIR}/Utils/CudaUtils.cu"
)
set_target_properties(cuda_utils PROPERTIES CUDA_ARCHITECTURES "80;70;60")


# Build Cpu Libraries
if (BUILD_CPU)
  # Combine Libaries
  add_library(libraries_cpu INTERFACE)
  target_link_libraries(
    libraries_cpu
    INTERFACE
    common_libraries
  )
endif()

# Build CUDA Libraries
if (CMAKE_CUDA_COMPILER AND BUILD_CUDA)
  # Combine Libaries
  add_library(
    libraries_cuda
    INTERFACE
  )
  target_link_libraries(
    libraries_cuda
    INTERFACE
    common_libraries
    cuda_utils
  )
endif()
