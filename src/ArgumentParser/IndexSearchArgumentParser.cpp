#include <iostream>
#include <memory>
#include <string>

#include "ArgumentParser/IndexSearchArgumentParser.h"
#include "cxxopts.hpp"

namespace sbwt_search {

using cxxopts::Options;
using cxxopts::ParseResult;
using cxxopts::value;
using std::make_unique;
using std::string;
using std::to_string;
using units_parser::MemoryUnitsParser;

IndexSearchArgumentParser::IndexSearchArgumentParser(
  const string &program_name,
  const string &program_description,
  int argc,
  char **argv
):
    options(
      make_unique<Options>(create_options(program_name, program_description))
    ),
    args{parse_arguments(argc, argv)} {}

auto IndexSearchArgumentParser::create_options(
  const string &program_name, const string &program_description
) -> Options {
  auto options = Options(program_name, program_description);
  options.add_options()("o,output-prefix", "Output prefix. The extension will be determined by the output format chosen", value<string>())(
      "i,index-file", "Index input file", value<string>()
    )("q,query-file",
      "The query in FASTA or FASTQ format, possibly gzipped."
      "Multi-line FASTQ is not supported. If the file extension "
      "is .txt, this is interpreted as a list of query files, one per line. "
      "In this case, --out-file is also interpreted as a list of output "
      "files "
      "in the same manner, one line for each input file.",
      value<string>()
    )("u,unavailable-main-memory",
      "The amount of main memory not to consume from the operating system in "
      "bits. This means that the program will hog as much main memory it "
      "can, provided that the VRAM can also keep up with it, except for the "
      "amount specified by this value. By default it is set to 4GB. The "
      "value can be in the following formats: 12345 (12345 bits), 12345B "
      "(12345 bytes), 12345KB, 12345MB or 12345GB",
      value<string>()->default_value(to_string(4ULL * 8 * 1024 * 1024 * 1024))
    )("m,max-main-memory",
      "The maximum amount of main memory (RAM) which may be used by the "
      "searching step, in bits. The default is that the program will occupy "
      "as much memory as it can, minus the unavailable main-memory. This "
      "value may be skipped by a few megabytes for its operation. It is only "
      "recommended to change this when you have a few small queries to "
      "process. The format of this value is the same as that for the "
      "unavailable-main-memory option",
      value<string>()->default_value(to_string(ULLONG_MAX))
    )("b,batches",
      "The number of batches to use. The default is 5. 1 is the minimum, and "
      "is equivalent to serial processing in terms of speed. This will split "
      "the main memory between the components. The more batches, the lower "
      "that a single batch's size. 5 is the recommended because there are 5 "
      "components so they can all keep processing without interruption from "
      "the start (this is assuming you have 5 threads running). If you have "
      "less threads, maybe set to to the number of available threads instead",
      value<u64>()->default_value(to_string(5))
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
  options.allow_unrecognised_options();
  return options;
}

auto IndexSearchArgumentParser::parse_arguments(int argc, char **argv)
  -> ParseResult {
  auto arguments = options->parse(argc, argv);
  if (argc == 1 || arguments["help"].as<bool>()) {
    std::cout << options->help() << std::endl;
    std::quick_exit(1);
  }
  return arguments;
}

auto IndexSearchArgumentParser::get_sequence_file() -> string {
  return args["query-file"].as<string>();
}
auto IndexSearchArgumentParser::get_index_file() -> string {
  return args["index-file"].as<string>();
}
auto IndexSearchArgumentParser::get_output_file() -> string {
  return args["output-prefix"].as<string>();
}
auto IndexSearchArgumentParser::get_unavailable_ram() -> u64 {
  return MemoryUnitsParser::convert(args["unavailable-main-memory"].as<string>()
  );
}
auto IndexSearchArgumentParser::get_max_cpu_memory() -> u64 {
  return MemoryUnitsParser::convert(args["max-main-memory"].as<string>());
}
auto IndexSearchArgumentParser::get_batches() -> u64 {
  return args["batches"].as<u64>();
}
auto IndexSearchArgumentParser::get_print_mode() -> string {
  return args["print-mode"].as<string>();
}

};  // namespace sbwt_search
