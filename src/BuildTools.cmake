# This file contains all the compilation necessary for modules I created
# to make development easier which can be used across projects not just
# for this one

include(ExternalProject)
include(FetchContent)

# External libraries used in the tools
# Fetch fmt
FetchContent_Declare(
  fmt
  QUIET
  GIT_REPOSITORY       https://github.com/fmtlib/fmt.git
  GIT_TAG              9.1.0
  GIT_SHALLOW          TRUE
)
FetchContent_MakeAvailable(fmt)

# Fetch spdlog
FetchContent_Declare(
  spdlog
  QUIET
  GIT_REPOSITORY       https://github.com/gabime/spdlog
  GIT_TAG              v1.10.0
  GIT_SHALLOW          TRUE
  SYSTEM
)
FetchContent_MakeAvailable(spdlog)
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include")

# My libraries
add_library(
  math_utils
  "${PROJECT_SOURCE_DIR}/Tools/MathUtils.cpp"
)
add_library(
  io_utils
  "${PROJECT_SOURCE_DIR}/Tools/IOUtils.cpp"
)
target_link_libraries(io_utils PRIVATE fmt::fmt)

add_library(
  error_utils
  "${PROJECT_SOURCE_DIR}/Tools/ErrorUtils.cpp"
)

add_library(
  logger
  "${PROJECT_SOURCE_DIR}/Tools/Logger.cpp"
)
target_link_libraries(logger PRIVATE spdlog::spdlog)

add_library(
  memory_units_parser
  "${PROJECT_SOURCE_DIR}/Tools/MemoryUnitsParser.cpp"
)

add_library(
  memory_utils
  "${PROJECT_SOURCE_DIR}/Tools/MemoryUtils.cpp"
)

add_library(
  omp_lock
  "${PROJECT_SOURCE_DIR}/Tools/OmpLock.cpp"
)
target_link_libraries(omp_lock PRIVATE OpenMP::OpenMP_CXX)

add_library(
  semaphore
  "${PROJECT_SOURCE_DIR}/Tools/Semaphore.cpp"
)
target_link_libraries(semaphore PRIVATE OpenMP::OpenMP_CXX omp_lock)

set(
  gpu_sources
  "${PROJECT_SOURCE_DIR}/Tools/GpuUtils.cu"
  "${PROJECT_SOURCE_DIR}/Tools/GpuPointer.cu"
  "${PROJECT_SOURCE_DIR}/Tools/GpuStream.cu"
  "${PROJECT_SOURCE_DIR}/Tools/GpuEvent.cu"
)

add_library(
  gpu_utils
  ${gpu_sources}
)
target_link_libraries(gpu_utils PUBLIC hip_rt)
set_source_files_properties(
  ${gpu_sources}
  TARGET_DIRECTORY gpu_utils
  PROPERTIES LANGUAGE ${HIP_TARGET_LANGUAGE}
)
