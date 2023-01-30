# Builds the main program, linking all the files and header files together

option(
  BUILD_MAIN
  "Build the main targets"
  ON
)

if (BUILD_MAIN)

add_library(
  index_search_main
  "${PROJECT_SOURCE_DIR}/Main/IndexSearchMain.cpp"
)
target_link_libraries(
  index_search_main
  PRIVATE
  common_libraries
)

add_executable(main "${PROJECT_SOURCE_DIR}/main.cpp")
target_link_libraries(
  main
  PRIVATE
  index_search_main
  common_libraries
)

endif()
