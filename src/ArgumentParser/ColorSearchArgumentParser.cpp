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
    "Input file which is the output of the previous step (index search). Note "
    "that we cannot directly use the output from the '--output-prefix' from "
    "the previous step since we must also have the file extension in this "
    "step. If the file extension is '.list', this is interpreted as a list of "
    "query files, one per line. In this case, --output-prefix must also be "
    "list of output files in the same manner, one line for each input file.",
    value<string>()
  );
  get_options().add_options()(
    "o,output-prefix",
    "The output file prefix or the output file list. If the file ends with the "
    "extension '.list', then it will be interepreted as a list of file output "
    "prefixes, separated by a newline. The extension of these output files "
    "will be determined by the choice of output format (look at the print-mode "
    "option for more information chosen",
    value<string>()
  );
  get_options().add_options()(
    "k,colors-file",
    "The *.tcolors file produced by themisto v3.0, which contains the "
    "colors data used in this program.",
    value<string>()
  );
  get_options().add_options()(
    "u,unavailable-main-memory",
    "The amount of main memory not to consume from the operating system in "
    "bits. This means that the program will hog as much main memory it "
    "can, provided that the VRAM (GPU memory) can also keep up with it, except "
    "for the amount specified by this value. By default it is set to 1GB. The "
    "value can be in the following formats: 12345B (12345 bytes), 12345KB, "
    "12345MB or 12345GB",
    value<string>()->default_value(to_string(gB_to_bits(1)))
  );
  get_options().add_options()(
    "m,max-main-memory",
    "The maximum amount of main memory (RAM) which may be used by the "
    "searching step, in bits. The default is that the program will occupy "
    "as much memory as it can, minus the unavailable main-memory. This "
    "value may be skipped by a few megabytes for its operation. It is only "
    "recommended to change this when you have a few small queries to "
    "process, so that intial memory allocation is faster. The format of this "
    "value is the same as that for the unavailable-main-memory option",
    value<string>()->default_value(to_string(ULLONG_MAX))
  );
  get_options().add_options()(
    "c,cpu-memory-percentage",
    "After calculating the memory usage using the formula: "
    "'min(system_max_memory, max-memory) - unavailable-max-memory', we "
    "multiply that value by memory-percentage, which is this parameter. This "
    "parameter is useful in case the program is unexpectedly taking too much "
    "memory. By default it is 0.8, which indicates that 80\% of available "
    "memory will be used. Note, the total memory used is not forced, and "
    "this is more of a soft maximum. The actual memory used will be slightly "
    "more for small variables and other registers used throughout the program.",
    value<double>()->default_value("0.8")
  );
  get_options().add_options()(
    "g,gpu-memory-percentage",
    "The percentage of gpu memory to use from the remaining free memory after "
    "the index has been loaded. This means that if we have 40GB of memory, and "
    "the index is 30GB, then we have 10GB left. If this value is set to 0.9, "
    "then 9GB will be used and the last 1GB of memory on the GPU will be left "
    "unused. The default value is 0.95, and unless you are running anything "
    "else on the machine which is also GPU heavy, it is recommended to leave "
    "it at this value.",
    value<double>()->default_value("0.95")
  );
  get_options().add_options()(
    "p,print-mode",
    "The mode used when printing the result to the output file. Options are "
    "'ascii' (default), 'binary' or 'csv'. In ascii omde, the results are "
    "printed in ASCII format so that the numbers viewed in each line represent "
    "the colors found. The reads are separated by newline characters. The "
    "binary format   will be similar to the index search in that each color "
    "found will be represented instead by an 8 byte (64-bit) integer, and the "
    "start of a new read is indicated by a -1 (ULLONG_MAX). This can result in "
    "larger files due to the characters taken by the colors usually being "
    "quite small, so the ascii format does not take as many characters. The "
    "csv format is the densest format and results in VERY huge files. As such "
    "it is only recommended to use it for smaller files. The format consists "
    "of comma separated 0s and 1s, where a 0 indicates that the color at that "
    "index has not been found, while a 1 represents the opposite.",
    value<string>()->default_value("ascii")
  );
  get_options().add_options()(
    "s,streams",
    "The number of files to read and write in parallel. This implies dividing "
    "the available memory into <memory>/<streams> pieces, so each batch will "
    "process less items at a time, but there is instead more parallelism. This "
    "should not be too high nor too large, as the number of threads spawned "
    "per file is already large, and it also depends on your disk drive. The "
    "default is 4. This means that 4 files will be processed at a time. If are "
    "processing less files than this, then the program will automatically "
    "default to using as many streams as you have files.",
    value<u64>()->default_value("4")
  );
  get_options().add_options()(
    "t,threshold",
    "The percentage of kmers within a read which need to be attributed to a "
    "color in order for us to accept that color as being part of our output. "
    "Must be a value between 1 and 0 (both included)",
    value<double>()->default_value("1")
  );
  get_options().add_options()(
    "include-not-found",
    "By default, indexes which have not been found in the index search "
    "(represented by -1s) are not considered by the algorithm, and they are "
    "simply skipped over and considered to not be part of the read. If this "
    "option is set, then they will be considered as indexes which have had no "
    "colors found."
  );
  get_options().add_options()(
    "include-invalid",
    "By default, indexes which are invalid, that is, the kmers to which they "
    "correspond to contain characters other than acgt/ACGT (represented by "
    "-2s) are not considered by the algorithm, and they are simply skipped "
    "over and considered to not be part of the read. If this option is set, "
    "then they will be considered as indexes which have had no colors found."
  );
  get_options().add_options()(
    "no-headers",
    "Do not write the headers to the outut files. The headers are the format "
    "name and version number written at the start of the file. The format of "
    "these 2 strings are that first we have a binary unsigned integer "
    "indicating the size of the string to come, followed by the string itself "
    "in ascii format, which contains as many bytes as indicated by the "
    "previous value. Then another unsigned integer indicating the size of the "
    "string representing the version number, and then the string in ascii "
    "bytes of the version number. For the csv version, this header is a comma "
    "separated list of color ids at the first line of the file. By default "
    "this option is false (meaning that the headers WILL be printed by "
    "default). Please note that if you wish to use the ascii or binary format "
    "for pseudoalignment later, this header is mandatory."
  );
  get_options().add_options()(
    "r,indexes-per-read",
    "The approximate number of indexes in every read. This is necessary "
    "because we need to keep track of the breaks where each read starts and "
    "ends in our list of base pairs. As such we must allocate memory for it. "
    "By defalt, this value is 70, meaning that we would then allocate enough "
    "memory for 1 break per 70 indexes. This option is available in case "
    "your reads vary a lot more than that and you wish to optimise for space.",
    value<u64>()->default_value("70")
  );
  get_options().add_options()(
    "h,help",
    "Print usage (you are here)",
    value<bool>()->default_value("false")
  );
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
  return get_args()["include-not-found"].as<bool>();
}
auto ColorSearchArgumentParser::get_include_invalid() const -> bool {
  return get_args()["include-invalid"].as<bool>();
}
auto ColorSearchArgumentParser::get_streams() const -> u64 {
  return get_args()["streams"].as<u64>();
}
auto ColorSearchArgumentParser::get_write_headers() const -> bool {
  return !get_args()["no-headers"].as<bool>();
}
auto ColorSearchArgumentParser::get_required_options() const -> vector<string> {
  return {
    "query-file",
    "colors-file",
    "output-prefix",
    "unavailable-main-memory",
    "max-main-memory",
    "streams",
    "print-mode"};
}

}  // namespace sbwt_search
