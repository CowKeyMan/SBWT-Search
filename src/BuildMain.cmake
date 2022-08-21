# Builds the main program, linking all the files and header files together

option(
  BUILD_MAIN
  "Build the main targets"
  ON
)

if (BUILD_MAIN)

# if (BUILD_CPU)
#   add_executable(main_cpu main.cpp)
#   target_link_libraries(
#     main_cpu
#     PRIVATE libraries_cpu
#   )
# endif()

if (BUILD_CUDA)
  add_executable(main_cuda main.cu)
  target_link_libraries(
    main_cuda
    PRIVATE
    libraries_cuda
    OpenMP::OpenMP_CXX
  )
set_target_properties(main_cuda PROPERTIES CUDA_ARCHITECTURES "60;70;80")
endif()

endif() # BUILD_MAIN
