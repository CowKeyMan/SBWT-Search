# Builds items which are commonly used between the main program and the tests.
# Usually these are classes and files which are used by the main program but
# are also tested individually

set(CMAKE_CXX_FLAGS_DEBUG_INIT "--coverage")

add_library(
  functions
  "${PROJECT_SOURCE_DIR}/Functions/Functions.cpp"
)
target_include_directories(functions PUBLIC ${PROJECT_SOURCE_DIR}/Functions)
target_link_libraries(functions PRIVATE common_options)

# Add more libraries here
