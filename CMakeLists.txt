# This is the root cmake file where the sources lie
# Different CMake versions have more features.
# Since we will be using CUDA, we want a pretty high version

cmake_minimum_required(VERSION 3.19)

project(
  "SBWT Search"
  VERSION 0.0.1
  DESCRIPTION "An application to search for k-mers in a genome given an SBWT index"
  HOMEPAGE_URL https://github.com/CowKeyMan/SBWT-Search
  LANGUAGES CXX
)

include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
endif()

include(cmake/ExportCompileCommands.cmake)
include(cmake/PreventBuildingInCmakeDirectory.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Options.cmake)
include(cmake/CCache.cmake)


add_subdirectory(src)
add_subdirectory (documentation)

include(cmake/StaticAnalyzers.cmake)
