# This is the root cmake file where the sources lie
# Different CMake versions have more features.
# Since we will be using CUDA, we want a pretty high version

cmake_minimum_required(VERSION 3.16)

project(
  CPP_github_workflow
  VERSION 1.0
  DESCRIPTION "A workflow base for any C++ project"
  HOMEPAGE_URL https://github.com/CowKeyMan/CPP_github_workflow
  LANGUAGES CXX
)

include(cmake/ExportCompileCommands.cmake)
include(cmake/PreventBuildingInCmakeDirectory.cmake)
include(cmake/CompilerWarnings.cmake)
include(cmake/Options.cmake)
include(cmake/CCache.cmake)
include(cmake/StaticAnalyzers.cmake)

add_subdirectory(src)
add_subdirectory (documentation)
