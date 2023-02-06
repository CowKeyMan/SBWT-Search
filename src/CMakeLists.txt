# Builds the main as well as testing programs, through the use of other .cmake
# files in this folder

# common options
add_compile_options(
  "$<$<CONFIG:Debug>:--coverage>" # only in debug mode
)
add_compile_options(
  "$<$<CONFIG:Release>:-O3>" # only in release mode
)
link_libraries(gcov)

project(tools)
include("${PROJECT_SOURCE_DIR}/BuildTools.cmake")

project(common)
include("${PROJECT_SOURCE_DIR}/BuildCommon.cmake")

project(src)
include("${PROJECT_SOURCE_DIR}/BuildMain.cmake")

project(test)
include("${PROJECT_SOURCE_DIR}/BuildTests.cmake")
