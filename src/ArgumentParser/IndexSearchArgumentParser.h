#ifndef INDEX_SEARCH_ARGUMENT_PARSER_H
#define INDEX_SEARCH_ARGUMENT_PARSER_H

/**
 * @file IndexSearchArgumentParser.h
 * @brief Contains functions to parse the main program's arguments
 */

#include <memory>
#include <string>

#include "ArgumentParser/ArgumentParser.h"
#include "Tools/MemoryUnitsParser.h"
#include "cxxopts.hpp"

namespace sbwt_search {

using cxxopts::Options;
using cxxopts::ParseResult;
using std::string;
using std::unique_ptr;
using units_parser::MemoryUnitsParser;

class IndexSearchArgumentParser: public ArgumentParser {
public:
  IndexSearchArgumentParser(
    const string &program_name,
    const string &program_description,
    int argc,
    char **argv
  );
  auto get_query_file() -> string;
  auto get_index_file() -> string;
  auto get_output_file() -> string;
  auto get_unavailable_ram() -> u64;
  auto get_max_cpu_memory() -> u64;
  auto get_batches() -> u64;
  auto get_print_mode() -> string;

private:
  auto create_options() -> void;
};

}  // namespace sbwt_search

#endif
