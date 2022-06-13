# Builds the main program, linking all the files and header files together

option(
  BUILD_MAIN
  "Build the main targets"
  ON
)

if (BUILD_MAIN)

if (BUILD_CPU)
  add_executable(main_cpu main.cpp)
  target_link_libraries(
    main_cpu
    PRIVATE common_options
    PRIVATE libraries_cpu
  )
  enable_warnings(main_cpu "PRIVATE")
endif()

if (BUILD_CUDA)
  add_executable(main_cuda main.cpp)
  target_link_libraries(
    main_cuda
    PRIVATE common_options
    PRIVATE libraries_cuda
  )
  set_target_properties(main_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80")
endif()

endif()
