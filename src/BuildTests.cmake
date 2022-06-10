# Builds the testing program. We use googletest as a testing framework

include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0
)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(test_lib INTERFACE)
target_link_libraries(
  test_lib INTERFACE
  gtest
  gtest_main
  gmock
  gmock_main
  gcov
)

set(
  functions_test_sources
  "${PROJECT_SOURCE_DIR}/Functions/Functions_test.cpp"
  # "${PROJECT_SOURCE_DIR}/others.cpp"
)

add_executable(
  test_main
  "${PROJECT_SOURCE_DIR}/test_main.cpp"
  "${functions_test_sources}"
  # "${other test sources}"
)
target_link_libraries(
  test_main
  PRIVATE
  test_lib
  functions
  # other libraries
)
target_compile_options(
  test_main PRIVATE
  "$<$<CONFIG:Debug>:--coverage>"
)
add_test(NAME test_main COMMAND test_main)
enable_warnings(test_main PRIVATE)
