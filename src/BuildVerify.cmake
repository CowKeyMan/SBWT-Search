# Builds the main program, linking all the files and header files together

option(
  BUILD_VERIFY
  "Build the verify target"
  ON
)


if (BUILD_VERIFY)
  add_executable(verify "${PROJECT_SOURCE_DIR}/verify.cpp")
  target_link_libraries(
    verify
    PRIVATE
    cxxopts
    output_parser
  )
endif()
