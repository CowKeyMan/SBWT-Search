#ifndef ARGUMENT_PARSER_H
#define ARGUMENT_PARSER_H

#include "cxxopts.hpp"

using cxxopts::value;
using std::string;

namespace sbwt_search {

const auto program_name = "SBWT Search";
const auto program_description
  = "An application to search for k-mers in a genome given an SBWT index";

auto parse_arguments(int argc, char **argv) -> cxxopts::ParseResult {
  cxxopts::Options options(program_name, program_description);
  options.add_options()(
    "o,out-file", "Output filename", value<string>())
    ("i,index-file", "Index input file", value<string>())
    (
      "q,query-file",
      "The query in FASTA or FASTQ format, possibly gzipped."
      "Multi-line FASTQ is not supported. If the file extension "
      "is .txt, this is interpreted as a list of query files, one per line. "
      "In this case, --out-file is also interpreted as a list of output files "
      "in the same manner, one line for each input file.",
      value<std::string>()
    )
    (
      "z,gzip-output",
      "Writes output in gzipped form. "
      "This can shrink the output files by an order of magnitude.",
      value<bool>()->default_value("false")
    )
    ("h,help", "Print usage", value<bool>()->default_value("false"))
  ;
  options.allow_unrecognised_options();
  auto arguments = options.parse(argc, argv);
  if (argc == 1 || arguments["help"].as<bool>()) {
    std::cout << options.help() << std::endl;
    exit(1);
  }
  return arguments;
}

}

#endif
