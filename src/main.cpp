#include <iostream>
#include <string>

#include "ArgumentParser.hpp"
#include "QueryReader.h"

using sbwt_search::parse_arguments;
using sbwt_search::QueryReader;
using std::string;

const auto kmer_size = 30;

auto main(int argc, char **argv) -> int {
  auto args = parse_arguments(argc, argv);
  auto query_reader = QueryReader(args["q"].as<string>(), kmer_size);
  query_reader.parse_kseqpp_streams();
}
