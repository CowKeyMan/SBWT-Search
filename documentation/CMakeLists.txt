# Configures the files related to doxygen as well as sphinx for RST documentation
# The actual building is done in a scrips/build/build_docs.sh

option(
  BUILD_DOCS
  "Build the documentations"
  ON
)

if (BUILD_DOCS)

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

endif()
