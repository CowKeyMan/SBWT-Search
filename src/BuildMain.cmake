# Builds the main program, linking all the files and header files together

option(
  BUILD_MAIN
  "Build the main targets"
  ON
)

if (BUILD_MAIN)

add_executable(main_cuda "${PROJECT_SOURCE_DIR}/main.cu")
target_link_libraries(
  main_cuda
  PRIVATE
  common_libraries
)
set_target_properties(main_cuda PROPERTIES CUDA_ARCHITECTURES "80;70;60")

endif()
