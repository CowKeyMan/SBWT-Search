#ifndef ARGUMENT_PARSER_HPP
#define ARGUMENT_PARSER_HPP

/**
 * @file ArgumentParser.hpp
 * @brief Contains functions to parse the main program's arguments
 * */

#include <memory>

#include "cxxopts.hpp"

using cxxopts::Options;
using cxxopts::ParseResult;
using cxxopts::value;
using std::make_unique;
using std::string;
using std::unique_ptr;

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
      options.add_options()("o,out-file", "Output filename", value<string>())(
        "i,index-file", "Index input file", value<string>()
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

    auto get_query_file() -> string { return args["q"].as<string>(); }

    auto get_index_file() -> string { return args["i"].as<string>(); }
};

}

#endif
