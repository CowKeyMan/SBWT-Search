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
  INTERFACE "$<$<CONFIG:Release>:-03>" # only in release mode
)
target_link_libraries(common_options INTERFACE gcov)

# Build Cpu Libraries
if (BUILD_CPU)
  add_library(
    functions_cpu
    "${PROJECT_SOURCE_DIR}/Functions/Functions_cpu.cpp"
  )
  target_include_directories(
    functions_cpu
    PUBLIC "${PROJECT_SOURCE_DIR}/Functions"
  )
  target_link_libraries(functions_cpu PRIVATE common_options)
  enable_warnings(functions_cpu "PRIVATE")

  # TODO: Add more cpu libraries here

  # Combine Libaries
  add_library(libraries_cpu INTERFACE)
  target_link_libraries(
    libraries_cpu
    INTERFACE functions_cpu
    # TODO: Combine more libraries that you create
  )
  target_link_libraries(libraries_cpu INTERFACE common_options)
endif()

# Build CUDA Libraries
if (CMAKE_CUDA_COMPILER AND BUILD_CUDA)
  add_library(
    functions_cuda
    "${PROJECT_SOURCE_DIR}/Functions/Functions_cuda.cu"
  )
  target_include_directories(
    functions_cuda
    PUBLIC "${PROJECT_SOURCE_DIR}/Functions"
    PUBLIC "${PROJECT_SOURCE_DIR}/Utils"
  )
  target_link_libraries(functions_cuda PRIVATE common_options)
  set_target_properties(functions_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80")

  # TODO: Add more cuda libraries here

  # Combine Libaries
  add_library(libraries_cuda INTERFACE)
  target_link_libraries(
    libraries_cuda
    INTERFACE functions_cuda
    # TODO: Combine more libraries that you create
  )
  target_link_libraries(functions_cuda INTERFACE common_options)
endif()
