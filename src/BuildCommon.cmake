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
## Fetch kseqpp
ExternalProject_Add(
  kseqpp
  GIT_REPOSITORY https://github.com/cartoonist/kseqpp
  GIT_TAG        v0.2.1
  PREFIX         "${CMAKE_BINARY_DIR}/external/kseqpp"
  CMAKE_ARGS
		-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/external/kseqpp/include")

## Fetch ZLIB
find_package(ZLIB)

## Fetch sdsl
ExternalProject_Add(
  sdsl
  GIT_REPOSITORY  https://github.com/simongog/sdsl-lite/
  GIT_TAG         v2.1.1
  PREFIX          "${CMAKE_BINARY_DIR}/external/sdsl"
  CMAKE_ARGS
    -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)
add_library(libsdsl SHARED IMPORTED)
set_target_properties(libsdsl PROPERTIES IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/external/sdsl/lib/libsdsl.a")
ExternalProject_Get_Property(sdsl SOURCE_DIR)
set (sdsl_SOURCE_DIR "${SOURCE_DIR}")
include_directories(SYSTEM "${sdsl_SOURCE_DIR}/include")

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
  sequence_file_parser
  "${PROJECT_SOURCE_DIR}/SequenceFileParser/SequenceFileParser.cpp"
)
add_library(
  sbwt_parser
  "${PROJECT_SOURCE_DIR}/SbwtParser/BitVectorSbwtParser.cpp"
  "${PROJECT_SOURCE_DIR}/SbwtParser/SdslSbwtParser.cpp"
)
target_link_libraries(sbwt_parser PRIVATE libsdsl)
add_library(
  sbwt_container_cpu
  "${PROJECT_SOURCE_DIR}/SbwtContainer/SbwtContainer.cpp"
)
target_link_libraries(sbwt_container_cpu PRIVATE libsdsl)
add_dependencies(sbwt_container_cpu sdsl)
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
set_target_properties(sbwt_container_gpu PROPERTIES CUDA_ARCHITECTURES "60;70;80")
add_library(sbwt_container INTERFACE)
target_link_libraries(
  sbwt_container
  INTERFACE
  sbwt_container_cpu
  sbwt_container_gpu
  libsdsl
)
add_library(
  sbwt_writer
  "${PROJECT_SOURCE_DIR}/SbwtWriter/SbwtWriter.cpp"
)
target_link_libraries(
  sbwt_writer
  PRIVATE
  libsdsl
  sbwt_container
)

# Common libraries
add_library(common_libraries INTERFACE)
target_link_libraries(
  common_libraries
  INTERFACE
  io_utils
  sequence_file_parser
  sbwt_parser
  libsdsl
  ZLIB::ZLIB
  parser
  cxxopts
  sbwt_container
  sbwt_writer
  OpenMP::OpenMP_CXX
  positions_builder
  # TODO: Link more libraries here
)
add_dependencies(common_libraries kseqpp)

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
    # TODO: Combine more libraries here which are cuda specific
  )
endif()
