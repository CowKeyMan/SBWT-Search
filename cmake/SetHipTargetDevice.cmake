option(
  HIP_TARGET_DEVICE
  "The target device of .cu files, traditionally cuda
  code. The value can be one of: NVIDIA, AMD, CPU. NVIDIA is chosen if the user
  wants to compile for an NVIDIA GPU, AMD is chosen if the user wishes to
  compiler for an AMD GPU and CPU is chosen if the user wants to instead compile
  to CPU, when they don't have a GPU. The default is NULL to make sure the user
  is intentional about their choice for this option."
  NULL
)
option(
  ROCM_BRANCH
  "The rocm version to choose. Set as the highest version at the time of
  writing by default"
  "rocm-5.4.x"
)

if(
    ${HIP_TARGET_DEVICE} STREQUAL "NVIDIA"
    OR
    ${HIP_TARGET_DEVICE} STREQUAL "AMD"
    OR
    ${HIP_TARGET_DEVICE} STREQUAL "CPU"
)
  find_program (BASH_EXECUTABLE bash REQUIRED)
  execute_process(
    COMMAND
    "${BASH_EXECUTABLE}" -c "${CMAKE_SOURCE_DIR}/scripts/configure/install_hip.sh ${HIP_TARGET_DEVICE} ${ROCM_BRANCH}"
  )
else()
  message(
    FATAL_ERROR
    "HIP_TARGET_DEVICE is not set to a proper target. Valid targets are:
    NVIDIA, AMD and CPU. See its help section for more details"
  )
endif()

add_library(hip_rt INTERFACE)

if(${HIP_TARGET_DEVICE} STREQUAL "NVIDIA")
  include(CheckLanguage)
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER)
    message("CUDA NOT FOUND")
  endif()
  enable_language(CUDA)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=all>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")
  target_include_directories(hip_rt SYSTEM INTERFACE "${CMAKE_BINARY_DIR}/hip/hip_nvidia/hipamd/build/include")
  set(HIP_TARGET_LANGUAGE CUDA)
endif()

if(${HIP_TARGET_DEVICE} STREQUAL "CPU")
  include(ExternalProject)
  include(FetchContent)
    FetchContent_Declare(
    hipcpu
    QUIET
    GIT_REPOSITORY       https://github.com/CowKeyMan/HIP-CPU.git
    GIT_TAG              92dd08ef2a735c4e8c230ead8f7e413eae99ed3f
    GIT_SHALLOW          TRUE
  )
  FetchContent_MakeAvailable(hipcpu)
  target_include_directories(hip_cpu_rt INTERFACE "${CMAKE_BINARY_DIR}/_deps/hipcpu-src/include/")
  target_compile_options(hip_rt INTERFACE -x c++)
  target_link_libraries(hip_rt INTERFACE hip_cpu_rt)
  set(HIP_TARGET_LANGUAGE CXX)
endif()