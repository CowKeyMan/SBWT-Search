#ifndef COLOR_SEARCH_ARGUMENT_PARSER_H
#define COLOR_SEARCH_ARGUMENT_PARSER_H

/**
 * @file ColorSearchArgumentParser.h
 * @brief Command line argument parser for the color searching / pseudoalignment
 * module
 */

#include <memory>
#include <string>

#include "ArgumentParser/ArgumentParser.h"
#include "Tools/TypeDefinitions.h"
#include "cxxopts.hpp"

namespace sbwt_search {

using cxxopts::Options;
using cxxopts::ParseResult;
using std::string;
using std::unique_ptr;

class ColorSearchArgumentParser: public ArgumentParser {
public:
  ColorSearchArgumentParser(
    const string &program_name,
    const string &program_description,
    int argc,
    char **argv
  );
  auto get_query_file() -> string;
  auto get_colors_file() -> string;
  auto get_output_file() -> string;
  auto get_unavailable_ram() -> u64;
  auto get_max_cpu_memory() -> u64;
  auto get_print_mode() -> string;
  auto get_batches() -> u64;

private:
  auto create_options() -> void;
};

}  // namespace sbwt_search

#endif
