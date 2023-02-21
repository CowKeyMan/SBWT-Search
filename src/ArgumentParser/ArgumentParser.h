#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

/**
 * @file ArgumentParser.h
 * @brief Parent class for argument parsers which take command line arguments
 * and parses them into usable data structures
 */

#include <memory>
#include <string>

#include "cxxopts.hpp"

namespace sbwt_search {

using cxxopts::Options;
using cxxopts::ParseResult;
using cxxopts::value;
using std::string;
using std::unique_ptr;

class ArgumentParser {
private:
  cxxopts::Options options;
  cxxopts::ParseResult args = {};

public:
  auto parse_arguments(int argc, char **argv) -> ParseResult;

  ArgumentParser(ArgumentParser &) = delete;
  ArgumentParser(ArgumentParser &&) = delete;
  auto operator=(ArgumentParser &) = delete;
  auto operator=(ArgumentParser &&) = delete;
  virtual ~ArgumentParser() = default;

protected:
  ArgumentParser(const string &program_name, const string &program_description);
  auto initialise_args(int argc, char **argv) -> void;
  [[nodiscard]] auto get_args() const -> const cxxopts::ParseResult &;
  auto get_options() -> cxxopts::Options &;
};

}  // namespace sbwt_search

#endif
