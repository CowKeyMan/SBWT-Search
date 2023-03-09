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
    ("b,batches",
      "The number of files to read and process at once. This implies dividing the available memory into <memory>/<batches> pieces, so each batch will process less items at a time, but there is instead more parallelism.",
      value<u64>()->default_value("4"))
     ("r,indexes-per-read", "The approximate number of indexes generated for every read. This is necessary because we need to store 'checkpoints' where each read starts and ends in our list of base pairs. As such we must allocate memory for it. By default, this value is 100, meaning that we would then allocate enough memory for 1 unisgned integer per 100 indexes. This option is available in case you need more memory than that.", value<u64>()->default_value("100"))
     ("g,gpu-memory-percentage", "The percentage of gpu memory to use from the remaining free memory after the index has been loaded. This means that if we have 40GB of memory, and the index is 30GB, then we have 10GB left. If this value is set to 0.9, then 9GB will be used and the last 1GB of memory on the GPU will be left unused. The default value is 0.95, and unless you are running anything else on the machine which is also GPU heavy, it is recommended to leave it at this value.", value<double>()->default_value("0.95"))
    ("p,cpu-memory-percentage", "After calculating the memory usage using the formula: 'min(system_max_memory, max-memory) - unavailable-max-memory', we multiply that value by memory-percentage, which is this parameter. This parameter is useful in case the program is unexpectedly taking too much memory. By default it is 0.8, which indicates that 80\% of available memory will be used. Note, the total memory used is not set in store, and this is more a minimum. The actual memory used will be slightly more for smal variables and other registers.", value<double>()->default_value("0.8"))
    ("c,print-mode",
      "The mode used when printing the result to the output file. Options: ascii",
      value<string>()->default_value("ascii")
    )
    ("t,threshold",
      "The percentage of kmers which need to be attributed to a color in order for us to accept that color as being part of our output. Must be a value between 1 and 0 (both included)",
      value<double>()->default_value("1"))
    ("no-ignore-not-found",
      "By default, indexes which have not been found in the sbwt (represented by -1s) are not considered by the algorithm, and they are simply skipped over and considered to not be part of the read. If this option is set, then they will be considered as reads which have had no colors found.")
    ("no-ignore-invalid",
      "By default, indexes which are invalid, that is, they contain characters other than acgt/ACGT (represented by -2s) are not considered by the algorithm, and they are simply skipped over and considered to not be part of the read. If this option is set, then they will be considered as reads which have had no colors found.")
    ("h,help", "Print usage", value<bool>()->default_value("false"));
  get_options().allow_unrecognised_options();
}

auto ColorSearchArgumentParser::get_query_file() const -> string {
  return get_args()["query-file"].as<string>();
}
auto ColorSearchArgumentParser::get_colors_file() const -> string {
  return get_args()["colors-file"].as<string>();
}
auto ColorSearchArgumentParser::get_output_file() const -> string {
  return get_args()["output-prefix"].as<string>();
}
auto ColorSearchArgumentParser::get_unavailable_ram() const -> u64 {
  return MemoryUnitsParser::convert(
    get_args()["unavailable-main-memory"].as<string>()
  );
}
auto ColorSearchArgumentParser::get_max_cpu_memory() const -> u64 {
  return MemoryUnitsParser::convert(get_args()["max-main-memory"].as<string>());
}
auto ColorSearchArgumentParser::get_print_mode() const -> string {
  return get_args()["print-mode"].as<string>();
}
auto ColorSearchArgumentParser::get_batches() const -> u64 {
  return get_args()["batches"].as<u64>();
}
auto ColorSearchArgumentParser::get_threshold() const -> double {
  auto threshold = get_args()["threshold"].as<double>();
  if (threshold < 0 || threshold > 1) {
    std::cerr
      << "Invalid value for threshold, must be between 1 and 0 (both included)"
      << std::endl;
    std::quick_exit(1);
  }
  return threshold;
}
auto ColorSearchArgumentParser::get_indexes_per_read() const -> u64 {
  return get_args()["indexes-per-read"].as<u64>();
}
auto ColorSearchArgumentParser::get_cpu_memory_percentage() const -> double {
  auto result = get_args()["cpu-memory-percentage"].as<double>();
  if (result < 0 || result > 1) {
    std::cerr
      << "Invalid value for cpu-memory-percentage. Must be between 0 and 1."
      << std::endl;
    std::quick_exit(1);
  }
  return result;
}
auto ColorSearchArgumentParser::get_gpu_memory_percentage() const -> double {
  auto result = get_args()["gpu-memory-percentage"].as<double>();
  if (result < 0 || result > 1) {
    std::cerr
      << "Invalid value for gpu-memory-percentage. Must be between 0 and 1."
      << std::endl;
    std::quick_exit(1);
  }
  return result;
}
auto ColorSearchArgumentParser::get_include_not_found() const -> bool {
  return get_args()["no-ignore-not-found"].as<bool>();
}
auto ColorSearchArgumentParser::get_include_invalid() const -> bool {
  return get_args()["no-ignore-invalid"].as<bool>();
}
auto ColorSearchArgumentParser::get_required_options() const -> vector<string> {
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
