#include <cmath>
#include <iostream>
#include <string>

#include "ArgumentParser/ArgumentParser.hpp"
#include "QueryFileParser/QueryFileParser.h"
#include "RawSequencesParser/RawSequencesParser.hpp"
#include "SbwtFactory/SbwtFactory.hpp"
#include "RankIndexBuilder/CpuRankIndexBuilder.hpp"

using sbwt_search::CharToBitsVector;
using sbwt_search::parse_arguments;
using sbwt_search::QueryFileParser;
using sbwt_search::RawSequencesParser;
using sbwt_search::SdslSbwtFactory;
using sbwt_search::CpuRankIndexBuilder;
using std::string;

const auto kmer_size = 30;
const size_t superblock_bits = 1024;
constexpr size_t hyperblock_bits = 1ULL << 32;  // pow(2, 32);

auto main(int argc, char **argv) -> int {
  auto args = parse_arguments(argc, argv);
  auto query_file_parser = QueryFileParser(args["q"].as<string>(), kmer_size);
  query_file_parser.parse_kseqpp_streams();
  auto sequences_parser = RawSequencesParser<CharToBitsVector>(
    query_file_parser.get_seqs(),
    query_file_parser.get_total_positions(),
    query_file_parser.get_total_letters(),
    kmer_size
  );
  sequences_parser.parse_serial();

  auto factory = SdslSbwtFactory();
  auto index_parser = factory.get_index_parser("path_to_sbwt_file");
  auto loaded_container = index_parser.parse();
  auto index_builder = CpuRankIndexBuilder<
    decltype(loaded_container),
    superblock_bits,
    hyperblock_bits>(loaded_container);
}
