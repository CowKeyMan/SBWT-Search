#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include "ArgumentParser/IndexSearchArgumentParser.h"
#include "Tools/MathUtils.hpp"
#include "cxxopts.hpp"

namespace sbwt_search {

using cxxopts::value;
using math_utils::gB_to_bits;
using std::string;
using std::to_string;
using units_parser::MemoryUnitsParser;

const u64 default_unavailable_gB = 4;

IndexSearchArgumentParser::IndexSearchArgumentParser(
  const string &program_name,
  const string &program_description,
  int argc,
  char **argv
):
    ArgumentParser::ArgumentParser(program_name, program_description) {
  create_options();
  initialise_args(argc, argv);
}

auto IndexSearchArgumentParser::create_options() -> void {
  get_options().add_options()("o,output-prefix", "Output prefix. The extension will be determined by the output format chosen", value<string>())(
      "i,index-file", "Index input file", value<string>()
    )("q,query-file",
      "The query in FASTA or FASTQ format, possibly gzipped."
      "Multi-line FASTQ is not supported. If the file extension "
      "is .txt, this is interpreted as a list of query files, one per line. "
      "In this case, --out-file is also interpreted as a list of output "
      "files "
      "in the same manner, one line for each input file.",
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
    )("m,max-main-memory",
      "The maximum amount of main memory (RAM) which may be used by the "
      "searching step, in bits. The default is that the program will occupy "
      "as much memory as it can, minus the unavailable main-memory. This "
      "value may be skipped by a few megabytes for its operation. It is only "
      "recommended to change this when you have a few small queries to "
      "process, so that intial memory allocation is faster. The format of this value is the same as that for the "
      "unavailable-main-memory option",
      value<string>()->default_value(to_string(ULLONG_MAX))
    )("s,streams",
      "The number of files to read and write in parallel. This implies dividing the available memory into <memory>/<streams> pieces, so each batch will process less items at a time, but there is instead more parallelism. This should not be too high nor too large, as the number of threads spawned per file is already large, and it also depends on your disk drive. The default is 4.",
      value<u64>()->default_value("4")
    )
    ("p,cpu-memory-percentage", "After calculating the memory usage using the formula: 'min(system_max_memory, max-memory) - unavailable-max-memory', we multiply that value by memory-percentage, which is this parameter. This parameter is useful in case the program is unexpectedly taking too much memory. By default it is 0.8, which indicates that 80\% of available memory will be used. Note, the total memory used is not set in store, and this is more a minimum. The actual memory used will be slightly more for smal variables and other registers.", value<double>()->default_value("0.8")
     )("r,base-pairs-per-read", "The approximate number of base pairs in every read. This is necessary because we need to store 'checkpoints' where each read starts and ends in our list of base pairs. As such we must allocate memory for it. By defalt, this value is 100, meaning that we would then allocate enough memory for 1 unisgned integer per 100 base pairs. This option is available in case you need more memory than that.", value<u64>()->default_value("100")
     )
     ("g,gpu-memory-percentage", "The percentage of gpu memory to use from the remaining free memory after the index has been loaded. This means that if we have 40GB of memory, and the index is 30GB, then we have 10GB left. If this value is set to 0.9, then 9GB will be used and the last 1GB of memory on the GPU will be left unused. The default value is 0.95, and unless you are running anything else on the machine which is also GPU heavy, it is recommended to leave it at this value.", value<double>()->default_value("0.95")
     )("c,print-mode",
      "The mode used when printing the result to the output file. Options "
      "are 'ascii' (default), 'binary' or 'boolean'. In ascii mode the "
      "results will be printed in ASCII format so that the number viewed "
      "output represents "
      "the position in the SBWT index. The outputs are separated by spaces "
      "and each word is separated by a newline. Strings which are not found "
      "are represented by -1 and strings which are invalid are represented "
      "by a -2. For binary format, the output is in binary. The numbers are "
      "placed in a single binary string where every 8 bytes represents an "
      "unsigned 64-bit number. Similarly to ASCII, strings which are not "
      "found are represented by a -1 (which loops around to become the "
      "maximum 64-bit integer (ULLONG_MAX=18446744073709551615)), strings "
      "which are invalid are represented by -2 (ULLONG_MAX-1) and strings "
      "are separeted by a -3 (ULLONG_MAX-2). The binary version is much "
      "faster but requires decoding the file later when it needs to be "
      "viewed. 'boolean' is the fastest mode however it is also the least "
      "desriptive. In this mode, 2 files are output. The first file is named "
      "by the given output file name, and contains 1 bit for each result. "
      "The string sizes are given in another file, where every 64 bit "
      "integer here is a string size. This is the fastest and most condensed "
      "way of printing the results, but we lose some information because we "
      "cannot say wether the result is invalid or just not found. At the end "
      "of this data file, the final number is padded by 0s to the next "
      "64-bit integer. Here, every 64 bit binary integer "
      "represents the amount of results for each string in the original "
      "input file. In terms of file extensions, ASCII format will add .txt, boolean format will add .bool and binary format will add .bin and .seqsizes for the separate sequence sizes",
      value<string>()->default_value("ascii")
    )("h,help", "Print usage", value<bool>()->default_value("false"));
  get_options().allow_unrecognised_options();
}

auto IndexSearchArgumentParser::get_query_file() const -> string {
  return get_args()["query-file"].as<string>();
}
auto IndexSearchArgumentParser::get_index_file() const -> string {
  return get_args()["index-file"].as<string>();
}
auto IndexSearchArgumentParser::get_output_file() const -> string {
  return get_args()["output-prefix"].as<string>();
}
auto IndexSearchArgumentParser::get_unavailable_ram() const -> u64 {
  return MemoryUnitsParser::convert(
    get_args()["unavailable-main-memory"].as<string>()
  );
}
auto IndexSearchArgumentParser::get_max_cpu_memory() const -> u64 {
  return MemoryUnitsParser::convert(get_args()["max-main-memory"].as<string>());
}
auto IndexSearchArgumentParser::get_streams() const -> u64 {
  return get_args()["streams"].as<u64>();
}
auto IndexSearchArgumentParser::get_print_mode() const -> string {
  return get_args()["print-mode"].as<string>();
}
auto IndexSearchArgumentParser::get_base_pairs_per_read() const -> u64 {
  return get_args()["base-pairs-per-read"].as<u64>();
}
auto IndexSearchArgumentParser::get_cpu_memory_percentage() const -> double {
  auto result = get_args()["cpu-memory-percentage"].as<double>();
  if (result < 0 || result > 1) {
    std::cerr
      << "Invalid value for cpu-memory-percentage. Must be between 0 and 1."
      << std::endl;
    std::quick_exit(1);
  }
  return result;
}
auto IndexSearchArgumentParser::get_gpu_memory_percentage() const -> double {
  auto result = get_args()["gpu-memory-percentage"].as<double>();
  if (result < 0 || result > 1) {
    std::cerr
      << "Invalid value for gpu-memory-percentage. Must be between 0 and 1."
      << std::endl;
    std::quick_exit(1);
  }
  return result;
}
auto IndexSearchArgumentParser::get_required_options() const -> vector<string> {
  return {
    "query-file",
    "index-file",
    "output-prefix",
    "max-main-memory",
    "unavailable-main-memory",
    "print-mode",
    "streams"};
}

};  // namespace sbwt_search
