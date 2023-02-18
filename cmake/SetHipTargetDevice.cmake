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

include(CheckLanguage)
if(${HIP_TARGET_DEVICE} STREQUAL "NVIDIA")
  check_language(CUDA)
  if (NOT CMAKE_CUDA_COMPILER)
    message("CUDA NOT FOUND")
  endif()
  enable_language(CUDA)
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--gpu-architecture=all>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Wno-deprecated-gpu-targets>")
  include_directories(SYSTEM "${CMAKE_BINARY_DIR}/hip/hip_nvidia/hipamd/build/include")
endif()
