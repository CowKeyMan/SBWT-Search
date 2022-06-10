# Builds the main program, linking all the files and header files together

add_executable(main main.cpp)

target_link_libraries(
  main
  PRIVATE
  functions
)
enable_warnings(main PRIVATE)
