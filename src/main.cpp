#include <iostream>
#include <string>

#include "ArgumentParser.hpp"
#include "QueryFileParser.h"
/* #include "RawSequencesParser.h" */

using sbwt_search::parse_arguments;
using sbwt_search::QueryFileParser;
using std::string;

auto main(int argc, char **argv) -> int {
  const auto kmer_size = 30;
  auto args = parse_arguments(argc, argv);
  auto query_file_parser = QueryFileParser(args["q"].as<string>(), kmer_size);
  query_file_parser.parse_kseqpp_streams();
  /* auto sequences_parser = RawSequencesParser( */
  /*   query_file_parser.get_seqs(), query_file_parser.get_total_positions() */
  /* ); */
  /* sequences_parser.parse(); */
}
