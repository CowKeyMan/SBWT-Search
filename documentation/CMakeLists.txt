# Looks for Doxygen and fails if it cannot find it. If it finds it, it will
# replace any text within the specified files (usually these files will end
# with the .in extension) which is surrounded by @ by the parameters provided
# by cmake. For example, if we look at the first line of
# documentation/index.rst, you will notice that part of the title is surrounded
# by @. This part will be replaced by the same value as it is within cmake. The
# output file is the same file but without the .in extension, so that the
# original file is unmodified. So if the input is a.txt.in, the output will be
# a.txt. Look at the configure_file cmake command here:
# https://cmake.org/cmake/help/latest/command/configure_file.html. This
# CMakeLists.txt is not to be used directly, but rather through the
# scripts/build/build_docs.sh script.

option(
  BUILD_DOCS
  "Build the documentations"
  ON
)

find_package(Doxygen)

if (DOXYGEN_FOUND)
  set(DOXYGEN_INPUT_DIR ${CMAKE_SOURCE_DIR}/src)
  set(DOXYGEN_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/build")

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile"
    @ONLY
  )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/index.rst.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/index.rst"
    @ONLY
  )

  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/conf.py"
    @ONLY
  )
else ()
    message("####################### ERROR ########################")
    message("Doxygen requested but executable not found")
    message("Read the documentation Tools section for installation instructions")
    message("######################################################\n")
    message(FATAL_ERROR "")
endif (DOXYGEN_FOUND)
