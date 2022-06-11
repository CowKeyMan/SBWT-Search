# Builds the main program, linking all the files and header files together

add_executable(main main.cpp)

# TODO: Link new libraries with main
target_link_libraries(
  main
  PRIVATE
  functions
)
target_link_libraries(main PUBLIC common_options)
