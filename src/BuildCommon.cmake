# Builds items which are commonly used between the main program and the tests.
# Usually these are classes, files and options which are used by the main program but
# are also tested individually

# Any common options are put as an interface
# rather than putting it with each file individually
add_library(common_options INTERFACE)
enable_warnings(common_options "INTERFACE")
target_compile_options(
  common_options INTERFACE
  "$<$<CONFIG:Debug>:--coverage>"
)
target_compile_options(
  common_options INTERFACE
  "$<$<CONFIG:Release>:-O3>"
)
target_link_libraries(
  common_options INTERFACE
  gcov
)

add_library(
  functions
  "${PROJECT_SOURCE_DIR}/Functions/Functions.cpp"
)
target_include_directories(functions PUBLIC ${PROJECT_SOURCE_DIR}/Functions)
target_link_libraries(functions PUBLIC common_options)

# TODO: Add more libraries here
