# Builds items which are commonly used between the main program and the tests.
# Usually these are classes, files and options which are used by the main program but
# are also tested individually

# Any common options are put as an interface
# rather than putting it with each file individually

add_library(common_options INTERFACE)
target_compile_options(
  common_options
  INTERFACE "$<$<CONFIG:Debug>:--coverage>" # only in debug mode
)
target_compile_options(
  common_options
  INTERFACE "$<$<CONFIG:Release>:-O3>" # only in release mode
)
target_link_libraries(common_options INTERFACE gcov)


include(ExternalProject)
ExternalProject_Add(
  kseqpp
  GIT_REPOSITORY https://github.com/cartoonist/kseqpp
  GIT_TAG        v0.2.1
  PREFIX         "${CMAKE_BINARY_DIR}/external/kseqpp"
  CMAKE_ARGS
		-DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)


# Query Reader Library
find_package(ZLIB)
add_library(
  query_reader
  "${PROJECT_SOURCE_DIR}/QueryReader/QueryReader.cpp"
)
target_link_libraries(
  query_reader
  PRIVATE
  common_options
  ZLIB::ZLIB
)
target_include_directories(
  query_reader
  PUBLIC "${PROJECT_SOURCE_DIR}/QueryReader"
  PUBLIC "${PROJECT_SOURCE_DIR}/Global"
  PRIVATE "${CMAKE_BINARY_DIR}/external/kseqpp/include"
)
add_dependencies(query_reader kseqpp)

# Common libraries
add_library(common_libraries INTERFACE)
target_link_libraries(
  common_libraries
  INTERFACE query_reader
)

# Build Cpu Libraries
if (BUILD_CPU)
  # Combine Libaries
  add_library(libraries_cpu INTERFACE)
  target_link_libraries(
    libraries_cpu
    INTERFACE common_options
    INTERFACE common_libraries
    # TODO: Combine more libraries that you create
  )
endif()

# Build CUDA Libraries
if (CMAKE_CUDA_COMPILER AND BUILD_CUDA)
  # Combine Libaries
  add_library(libraries_cuda INTERFACE)
  target_link_libraries(
    libraries_cuda
    INTERFACE common_options
    INTERFACE common_libraries
    # TODO: Combine more libraries that you create
  )
endif()
