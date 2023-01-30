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
target_link_libraries(index_search_main PRIVATE common_libraries)

add_library(
  color_search_main
  "${PROJECT_SOURCE_DIR}/Main/ColorSearchMain.cpp"
)
target_link_libraries(color_search_main PRIVATE common_libraries)

add_executable(sbwt_search "${PROJECT_SOURCE_DIR}/main.cpp")
target_link_libraries(
  sbwt_search
  PRIVATE
  index_search_main
  color_search_main
  common_libraries
)

endif()
