# Builds the main program, linking all the files and header files together

option(
  BUILD_MAIN
  "Build the main targets"
  ON
)


# Common Libraries
## Fetch cxxopts
include(FetchContent)
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
FetchContent_MakeAvailable(cxxopts)

if (BUILD_MAIN)
add_library(libraries_main INTERFACE)
target_link_libraries(
  libraries_main
  INTERFACE cxxopts
  INTERFACE common_options
)

if (BUILD_CPU)
  add_executable(main_cpu main.cpp)
  target_link_libraries(
    main_cpu
    PRIVATE libraries_main
    PRIVATE libraries_cpu
  )
  enable_warnings(main_cpu "PRIVATE")
endif()

if (BUILD_CUDA)
  add_executable(main_cuda main.cpp)
  target_link_libraries(
    main_cuda
    PRIVATE libraries_main
    PRIVATE libraries_cuda
  )
  set_target_properties(main_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80")
endif()

endif() # BUILD_MAIN
