# Builds the main program, linking all the files and header files together

option(
  BUILD_MAIN
  "Build the main targets"
  ON
)

if (BUILD_MAIN)

add_executable(main "${PROJECT_SOURCE_DIR}/main.cu")
target_link_libraries(
  main
  PRIVATE
  common_libraries
)
set_target_properties(main PROPERTIES CUDA_ARCHITECTURES "80;70;60")

endif()
