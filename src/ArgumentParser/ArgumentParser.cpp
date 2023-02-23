#include <iostream>
#include <memory>

#include "ArgumentParser/ArgumentParser.h"
#include "cxxopts.hpp"

namespace sbwt_search {

using std::cout;
using std::endl;
using std::make_unique;

ArgumentParser::ArgumentParser(
  const string &program_name, const string &program_description
):
    options(program_name, program_description) {}

auto ArgumentParser::parse_arguments(int argc, char **argv) -> ParseResult {
  auto arguments = options.parse(argc, argv);
  if (arguments["help"].as<bool>() || !is_required_all_provided(arguments)) {
    cout << options.help() << endl;
    std::quick_exit(1);
  }
  return arguments;
}

auto ArgumentParser::initialise_args(int argc, char **argv) -> void {
  args = parse_arguments(argc, argv);
}
auto ArgumentParser::get_args() const -> const cxxopts::ParseResult & {
  return args;
}
auto ArgumentParser::get_options() -> cxxopts::Options & { return options; }
auto ArgumentParser::is_required_all_provided(ParseResult &args) -> bool {
  for (auto &s : get_required_options()) {
    if (args[s].count() < 1 && !args[s].has_default()) {
      cout << "Missing option: " << s << endl;
      return false;
    }
  }
  return true;
}

}  // namespace sbwt_search
