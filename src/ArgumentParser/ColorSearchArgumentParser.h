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
  auto get_query_file() const -> string;
  auto get_colors_file() const -> string;
  auto get_output_file() const -> string;
  auto get_unavailable_ram() const -> u64;
  auto get_max_cpu_memory() const -> u64;
  auto get_print_mode() const -> string;
  auto get_threshold() const -> double;
  auto get_indexes_per_read() const -> u64;
  auto get_cpu_memory_percentage() const -> double;
  auto get_gpu_memory_percentage() const -> double;
  auto get_include_not_found() const -> bool;
  auto get_include_invalid() const -> bool;
  auto get_streams() const -> u64;
  auto get_write_headers() const -> bool;

private:
  auto create_options() -> void;

protected:
  auto get_required_options() const -> vector<string> override;
};

}  // namespace sbwt_search

#endif
