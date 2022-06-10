# Builds the main as well as testing programs

project(common)
include("${PROJECT_SOURCE_DIR}/BuildCommon.cmake")

project(src)
include("${PROJECT_SOURCE_DIR}/BuildMain.cmake")

project(test)
include("${PROJECT_SOURCE_DIR}/BuildTests.cmake")
