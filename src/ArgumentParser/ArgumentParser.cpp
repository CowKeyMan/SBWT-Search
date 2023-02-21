#include <memory>

#include "ArgumentParser/ArgumentParser.h"

namespace sbwt_search {

using cxxopts::Options;
using std::make_unique;

ArgumentParser::ArgumentParser(
  const string &program_name, const string &program_description
):
    options(program_name, program_description) {}

auto ArgumentParser::parse_arguments(int argc, char **argv) -> ParseResult {
  auto arguments = options.parse(argc, argv);
  if (argc == 1 || arguments["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
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

}  // namespace sbwt_search
