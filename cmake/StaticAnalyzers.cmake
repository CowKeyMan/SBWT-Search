# Static analyzers check the code is in a consistent and clean format before compiling
# For checking, some of them run python scripts which have been created for the purpose
# of running the tools or are new tools themselves (such as the header guard check)

option(ENABLE_CLANG_TIDY "Enable static analysis with clang-tidy" ON)
option(ENABLE_HEADER_GUARDS_CHECK "Enable header guards check" ON)
option(ENABLE_CLANG_FORMAT_CHECK "Enable header clang-format check" ON)

if(ENABLE_CLANG_TIDY)
  find_program(CLANGTIDY clang-tidy)
  if(CLANGTIDY)
    message("\nRunning clang-tidy checks...")
    message("********************************************************")
    execute_process(
      COMMAND python scripts/cmake_checks/clang_tidy.py
      WORKING_DIRECTORY  ${CMAKE_SOURCE_DIR}
    )
  else()
    message("####################### ERROR ########################")
    message("clang-tidy requested but executable not found")
    message("Read the documentation Tools section for installation instructions")
    message("######################################################\n")
    message(FATAL_ERROR "")
  endif()
endif()

if(ENABLE_HEADER_GUARDS_CHECK)
  message("\nRunning header checks...")
  message("********************************************************")
  execute_process(
    COMMAND python3 scripts/cmake_checks/check_headers.py
    WORKING_DIRECTORY  ${CMAKE_SOURCE_DIR}
  )
  message("")
endif()

if(ENABLE_CLANG_FORMAT_CHECK)
  find_program(CLANGFORMAT clang-format)
  if(CLANGFORMAT)
    message("\nRunning clang-format checks...")
    message("********************************************************")
    execute_process(
      COMMAND python scripts/cmake_checks/check_clang_format.py
      WORKING_DIRECTORY  ${CMAKE_SOURCE_DIR}
    )
  else()
    message("####################### ERROR ########################")
    message("clang-format requested but executable not found")
    message("Read the documentation Tools section for installation instructions")
    message("######################################################\n")
    message(FATAL_ERROR "")
  endif()
endif()
