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

if(EXISTS "${CMAKE_BINARY_DIR}/external/kseqpp/lib/pkgconfig/kseq++.pc")
  set(KSEQPP_FOUND TRUE)
else()
  set(KSEQPP_FOUND FALSE)
endif()
if (NOT KSEQPP_FOUND)
## Fetch kseqpp
ExternalProject_Add(
  kseqpp
  GIT_REPOSITORY https://github.com/cartoonist/kseqpp
  GIT_TAG        v0.2.1
  PREFIX         "${CMAKE_BINARY_DIR}/external/kseqpp"
  CMAKE_ARGS
		-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
endif()
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/external/kseqpp/include")

## Fetch ZLIB
find_package(ZLIB)

find_library(SDSL_FOUND NAMES libsdsl sdsl PATHS "${CMAKE_BINARY_DIR}/external/sdsl/lib/" "${CMAKE_BINARY_DIR}/external/sdsl/")
if (NOT SDSL_FOUND)
## Fetch sdsl
  ExternalProject_Add(
    sdsl
    GIT_REPOSITORY  https://github.com/simongog/sdsl-lite/
    GIT_TAG         v2.1.1
    PREFIX          "${CMAKE_BINARY_DIR}/external/sdsl"
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
  )
endif()
add_library(libsdsl SHARED IMPORTED)
set_target_properties(libsdsl PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external/sdsl/lib/libsdsl.a")
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/external/sdsl/include")


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
  parser
  "${PROJECT_SOURCE_DIR}/Builder/Builder.cpp"
)
add_library(
  io_utils
  "${PROJECT_SOURCE_DIR}/Utils/IOUtils.cpp"
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
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/SequenceFileParser.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/ContinuousSequenceFileParser.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/IntervalBatchProducer.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/CumulativePropertiesBatchProducer.cpp"
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/StringSequenceBatchProducer.cpp"
)
target_link_libraries(sequence_file_parser PRIVATE io_utils fmt::fmt)
if(NOT KSEQPP_FOUND)
add_dependencies(sequence_file_parser kseqpp)
endif()
add_library(
  filenames_parser
  "${PROJECT_SOURCE_DIR}/FilenamesParser/FilenamesParser.cpp"
)
add_library(
  sbwt_builder
  "${PROJECT_SOURCE_DIR}/SbwtBuilder/SbwtBuilder.cpp"
)
target_link_libraries(sbwt_builder PRIVATE libsdsl io_utils)
if (NOT SDSL_FOUND)
add_dependencies(sbwt_builder sdsl)
endif()
add_library(
  sbwt_container_cpu
  "${PROJECT_SOURCE_DIR}/SbwtContainer/SbwtContainer.cpp"
)
if (NOT SDSL_FOUND)
add_dependencies(sbwt_container_cpu sdsl)
endif()
target_link_libraries(sbwt_container_cpu PRIVATE libsdsl)
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
if(NOT KSEQPP_FOUND)
add_dependencies(sbwt_container_gpu kseqpp)
endif()
if (NOT SDSL_FOUND)
add_dependencies(sbwt_container_gpu sdsl)
endif()
add_library(sbwt_container INTERFACE)
target_link_libraries(
  sbwt_container
  INTERFACE
  sbwt_container_cpu
  sbwt_container_gpu
  libsdsl
)

# Common libraries
add_library(common_libraries INTERFACE)
target_link_libraries(
  common_libraries
  INTERFACE
  io_utils
  sequence_file_parser
  filenames_parser
  sbwt_builder
  libsdsl
  ZLIB::ZLIB
  parser
  cxxopts
  sbwt_container
  positions_builder
  OpenMP::OpenMP_CXX
  logger
  fmt::fmt
  error_utils
  # TODO: Link more libraries here
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
    # TODO: Combine more libraries here which are cpu specific
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
    # TODO: Combine more libraries here which are cuda specific
  )
endif()
