# This file contains all the compilation necessary for modules I created
# to make development easier which can be used across projects not just
# for this one

include(ExternalProject)
include(FetchContent)

# External libraries used in the tools
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
  )
FetchContent_MakeAvailable(spdlog)
include_directories(SYSTEM "${CMAKE_BINARY_DIR}/_deps/spdlog-src/include")

# My libraries
add_library(
  io_utils
  "${PROJECT_SOURCE_DIR}/Tools/IOUtils.cpp"
)
target_link_libraries(
  io_utils
  PRIVATE
  fmt::fmt
)

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

add_library(
  semaphore
  "${PROJECT_SOURCE_DIR}/Tools/Semaphore.cpp"
)
target_link_libraries(semaphore PRIVATE OpenMP::OpenMP_CXX omp_lock)

add_library(
  cuda_utils
  "${PROJECT_SOURCE_DIR}/Tools/CudaUtils.cu"
  "${PROJECT_SOURCE_DIR}/Tools/CudaPointer.cu"
)
set_target_properties(cuda_utils PROPERTIES CUDA_ARCHITECTURES "80;70;60")
