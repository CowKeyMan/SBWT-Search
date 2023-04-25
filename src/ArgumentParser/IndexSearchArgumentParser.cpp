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
  get_options().add_options()(
    "q,query-file",
    "The query in FASTA or FASTQ format, possibly gzipped, and also possibly a "
    "combination of both. Empty lines are also supported. "
    "If the file extension is '.list', this is interpreted as a list of query "
    "files, one per line. In this case, --output-prefix must also be "
    "list of output files in the same manner, one line for each input file.",
    value<string>()
  );
  get_options().add_options()(
    "i,index-file",
    "The themisto *.tdbg file or SBWT's *.sbwt file. The program is compatible "
    "with both. This contains the 4 bit vectors for acgt as well as the k for "
    "the k-mers used.",
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
    "r,base-pairs-per-seq",
    "The approximate number of base pairs in every seq. This is necessary "
    "because we need to keep track of the breaks where each seq starts and "
    "ends in our list of base pairs. As such we must allocate memory for it. "
    "By defalt, this value is 100, meaning that we would then allocate enough "
    "memory for 1 break per 100 base pairs. This option is available in case "
    "your seqs vary a lot more than that and you wish to optimise for space.",
    value<u64>()->default_value("100")
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
    "p,print-mode",
    "The mode used when printing the result to the output file. Options "
    "are 'ascii' (default), 'binary' or 'bool'. In ascii mode the "
    "results will be printed in ASCII format so that the number viewed "
    "represents the position in the SBWT index. The indexes within a seq are "
    "separated by spaces and each seq is separated by a newline. Strings "
    "which are not found are represented by -1 and strings which are invalid "
    "(they contain characters other than ACGT) are represented by a -2. For "
    "binary format, the output is in binary, that is, each index takes 8 bits. "
    "The numbers are placed in a single binary string where every 8 bytes "
    "represents an unsigned 64-bit number. Similarly to ASCII, strings which "
    "are not found are represented by a -1 (which loops around to become the "
    "maximum 64-bit integer (ULLONG_MAX=18446744073709551615)), strings which "
    "are invalid are represented by -2 (ULLONG_MAX-1) and seqs are separeted "
    "by a -3 (ULLONG_MAX-2). This version turns out to be slower and uses more "
    "space, it is only recommended if your indexes are huge (mostly larger "
    "than 8 bits). 'bool' is the fastest mode however it is also the least "
    "desriptive. In this mode, each index results in a single ASCII byte, "
    "which contains the value 0 if found, 1 if not found and 2 if the value is "
    "invalid. Similarly to the ascii format, each seq is separated by a "
    "newline. This is the fastest and most condensed way of printing the "
    "results, but we lose the position in the index, and therefore we cannot "
    "use this format for pseudoalignment. In terms of file extensions, ASCII "
    "format will add '.txt', boolean format will add '.bool' and binary format "
    "will add '.bin'.",
    value<string>()->default_value("ascii")
  );
  get_options().add_options()(
    "k,colors-file",
    "The *.tcolors file produced by themisto v3.0, which contains the "
    "key_kmer_marks as one of its components within, used in this program. "
    "When this option is given, then the index search will move to the next "
    "key kmer. If not given, then the program will simply get the index of the "
    "node at which the given k-mer lands on.",
    value<string>()->default_value("")
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
    "bytes of the version number. By default this option is false (meaning "
    "that the headers WILL be printed by default). Please note that if you "
    "wish to use the ascii or binary format for pseudoalignment later, this "
    "header is mandatory. "
  );
  get_options().add_options()(
    "h,help",
    "Print usage (you are here)",
    value<bool>()->default_value("false")
  );
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
auto IndexSearchArgumentParser::get_print_mode() const -> string {
  return get_args()["print-mode"].as<string>();
}
auto IndexSearchArgumentParser::get_base_pairs_per_seq() const -> u64 {
  return get_args()["base-pairs-per-seq"].as<u64>();
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
auto IndexSearchArgumentParser::get_streams() const -> u64 {
  return get_args()["streams"].as<u64>();
}
auto IndexSearchArgumentParser::get_colors_file() const -> string {
  return get_args()["colors-file"].as<string>();
}
auto IndexSearchArgumentParser::get_write_headers() const -> bool {
  return !get_args()["no-headers"].as<bool>();
}
auto IndexSearchArgumentParser::get_required_options() const -> vector<string> {
  return {
    "query-file",
    "index-file",
    "output-prefix",
    "max-main-memory",
    "unavailable-main-memory",
    "streams",
    "print-mode"};
}

};  // namespace sbwt_search
