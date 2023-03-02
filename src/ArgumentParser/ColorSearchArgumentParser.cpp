#include <iostream>
#include <limits>
#include <string>

#include "ArgumentParser/ColorSearchArgumentParser.h"
#include "Tools/MathUtils.hpp"
#include "Tools/MemoryUnitsParser.h"

namespace sbwt_search {

using math_utils::gB_to_bits;
using std::numeric_limits;
using std::to_string;
using units_parser::MemoryUnitsParser;

const u64 default_unavailable_gB = 4;
const u64 default_batches = 1;

ColorSearchArgumentParser::ColorSearchArgumentParser(
  const string &program_name,
  const string &program_description,
  int argc,
  char **argv
):
    ArgumentParser::ArgumentParser(program_name, program_description) {
  create_options();
  initialise_args(argc, argv);
}

auto ColorSearchArgumentParser::create_options() -> void {
  get_options().add_options()(
    "q,query-file",
    "Input file which is the output of the previous step (index search)",
    value<string>()
  )
    (
    "i,colors-file",
    "The colors index file",
    value<string>()
    )
    (
    "o,output-prefix",
    "The prefix for the output file",
    value<string>()
    )
    ("u,unavailable-main-memory",
      "The amount of main memory not to consume from the operating system in "
      "bits. This means that the program will hog as much main memory it "
      "can, provided that the VRAM can also keep up with it, except for the "
      "amount specified by this value. By default it is set to 4GB. The "
      "value can be in the following formats: 12345 (12345 bits), 12345B "
      "(12345 bytes), 12345KB, 12345MB or 12345GB",
      value<string>()->default_value(to_string(gB_to_bits(default_unavailable_gB)))
    )
    ("m,max-main-memory",
      "The maximum amount of main memory (RAM) which may be used by the "
      "searching step, in bits. The default is that the program will occupy "
      "as much memory as it can, minus the unavailable main-memory. This "
      "value may be skipped by a few megabytes for its operation. It is only "
      "recommended to change this when you have a few small queries to "
      "process. The format of this value is the same as that for the "
      "unavailable-main-memory option",
      value<string>()->default_value(to_string(numeric_limits<u64>::max()))
     )
    ("c,print-mode",
      "The mode used when printing the result to the output file. Options: ascii",
      value<string>()->default_value("ascii")
    )
    ("b,batches",
      "The number of files to read and process at once. This implies dividing the available memory into <memory>/<batches> pieces, so each batch will process less items at a time, but there is instead more parallelism.",
      value<u64>()->default_value(to_string(default_batches)))
    ("t,threshold",
      "The percentage of kmers which need to be attributed to a color in order for us to accept that color as being part of our output. Must be a value between 1 and 0 (both included)",
      value<double>()->default_value("1")
    )("h,help", "Print usage", value<bool>()->default_value("false"));
  get_options().allow_unrecognised_options();
}

auto ColorSearchArgumentParser::get_query_file() -> string {
  return get_args()["query-file"].as<string>();
}
auto ColorSearchArgumentParser::get_colors_file() -> string {
  return get_args()["colors-file"].as<string>();
}
auto ColorSearchArgumentParser::get_output_file() -> string {
  return get_args()["output-prefix"].as<string>();
}
auto ColorSearchArgumentParser::get_unavailable_ram() -> u64 {
  return MemoryUnitsParser::convert(
    get_args()["unavailable-main-memory"].as<string>()
  );
}
auto ColorSearchArgumentParser::get_max_cpu_memory() -> u64 {
  return MemoryUnitsParser::convert(get_args()["max-main-memory"].as<string>());
}
auto ColorSearchArgumentParser::get_print_mode() -> string {
  return get_args()["print-mode"].as<string>();
}
auto ColorSearchArgumentParser::get_batches() -> u64 {
  return get_args()["batches"].as<u64>();
}
auto ColorSearchArgumentParser::get_threshold() -> double {
  auto threshold = get_args()["threshold"].as<double>();
  if (threshold < 0 || threshold > 1) {
    std::cerr
      << "Invalid value for threshold, must be between 1 and 0 (both included)"
      << std::endl;
    std::quick_exit(1);
  }
  return threshold;
}
auto ColorSearchArgumentParser::get_required_options() -> vector<string> {
  return {
    "query-file",
    "colors-file",
    "output-prefix",
    "unavailable-main-memory",
    "max-main-memory",
    "print-mode",
    "batches"};
}

}  // namespace sbwt_search
