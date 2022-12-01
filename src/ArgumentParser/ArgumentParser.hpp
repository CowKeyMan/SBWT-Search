#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP

/**
 * @file ArgumentParser.hpp
 * @brief Contains functions to parse the main program's arguments
 * */

#include <climits>
#include <memory>
#include <string>

#include "Utils/MemoryUnitsParser.h"
#include "cxxopts.hpp"

using cxxopts::Options;
using cxxopts::ParseResult;
using cxxopts::value;
using std::make_unique;
using std::string;
using std::to_string;
using std::unique_ptr;
using units_parser::MemoryUnitsParser;

namespace sbwt_search {

class ArgumentParser {
  private:
    unique_ptr<cxxopts::Options> options;
    cxxopts::ParseResult args;

  public:
    ArgumentParser(
      const string program_name,
      const string program_description,
      int argc,
      char **argv
    ) {
      options
        = make_unique<Options>(create_options(program_name, program_description)
        );
      args = parse_arguments(argc, argv);
    }

    auto
    create_options(const string program_name, const string program_description)
      -> Options {
      Options options = Options(program_name, program_description);
      options.add_options()(
        "o,output-file", "Output filename", value<string>()
      )("i,index-file", "Index input file", value<string>()
      )("q,query-file",
        "The query in FASTA or FASTQ format, possibly gzipped."
        "Multi-line FASTQ is not supported. If the file extension "
        "is .txt, this is interpreted as a list of query files, one per line. "
        "In this case, --out-file is also interpreted as a list of output "
        "files "
        "in the same manner, one line for each input file.",
        value<string>()
      )("z,gzip-output",
        "Writes output in gzipped form. "
        "This can shrink the output files by an order of magnitude.",
        value<bool>()->default_value("false")
      )("u,unavailable-main-memory",
        "The amount of main memory not to consume from the operating system in "
        "bits. This means that the program will hog as much main memory it "
        "can, provided that the VRAM can also keep up with it, except for the "
        "amount specified by this value. By default it is set to 4GB",
        value<string>()->default_value(to_string(4ULL * 8 * 1024 * 1024 * 1024))
      )("m,max-main-memory",
        "The maximum amount of main memory (RAM) which may be used by the "
        "searching step, in bits. The default is that the program will occupy "
        "as much memory as it can, minus the unavailable main-memory. This "
        "value may be skipped by a few megabytes for its operation. It is only "
        "recommended to change this when you have a few small queries to "
        "process.",
        value<string>()->default_value(to_string(ULLONG_MAX))
      )("b,batches",
        "The number of batches to use. The default is 2. 1 is the minimum, and "
        "is equivalent to serial processing in terms of speed. The more "
        "batches, the more multiprocessing may be available, but if one step "
        "of the pipeline is much slower, then this may be a bottleneck to the "
        "rest of the steps",
        value<unsigned int>()->default_value(to_string(2))
      )("c,print-mode",
        "The mode used when printing the result to the output file. Options "
        "are 'ascii' (default), 'binary' or 'boolean'. In ascii mode the "
        "results will be "
        "printed in ASCII format so that the number viewed output represents "
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
        "desriptive. In this mode the first unsigned 64-bit number (in binary) "
        "represents the length of the string, followed by a series of binary "
        "bits which tell if the kmer was found or not. If it was found, a 1 is "
        "in that place, and if it was not found or the kmer is invalid, then a "
        "0 is in that place. The binary bits are rounded up to the next "
        "multiple of 8, so that the user may read the output file character by "
        "character. This cycle then continues so that the length of the next "
        "string is written followed by the binary boolean bit_vector of that "
        "string padded to the next multiple of 8.",
        value<string>()->default_value("ascii")
      )("h,help", "Print usage", value<bool>()->default_value("false"));
      options.allow_unrecognised_options();
      return options;
    }

    auto parse_arguments(int argc, char **argv) -> ParseResult {
      auto arguments = options->parse(argc, argv);
      if (argc == 1 || arguments["help"].as<bool>()) {
        std::cout << options->help() << std::endl;
        exit(1);
      }
      return arguments;
    }

    auto get_sequence_file() -> string {
      return args["query-file"].as<string>();
    }
    auto get_index_file() -> string { return args["index-file"].as<string>(); }
    auto get_output_file() -> string {
      return args["output-file"].as<string>();
    }
    auto get_unavailable_ram() -> size_t {
      return MemoryUnitsParser::convert(
        args["unavailable-main-memory"].as<string>()
      );
    }
    auto get_max_cpu_memory() -> size_t {
      return MemoryUnitsParser::convert(args["max-main-memory"].as<string>());
    }
    auto get_batches() -> unsigned int {
      return args["batches"].as<unsigned int>();
    }
    auto get_print_mode() -> string { return args["print-mode"].as<string>(); }
};

}  // namespace sbwt_search

#endif
