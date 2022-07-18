#include <cmath>
#include <iostream>
#include <string>

#include "ArgumentParser/ArgumentParser.hpp"
#include "Presearcher/Presearcher.h"
#include "QueryFileParser/QueryFileParser.h"
#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "RawSequencesParser/RawSequencesParser.hpp"
#include "SbwtFactory/SbwtFactory.hpp"

using sbwt_search::BitVectorSbwtFactory;
using sbwt_search::CharToBitsVector;
using sbwt_search::CpuRankIndexBuilder;
using sbwt_search::parse_arguments;
using sbwt_search::Presearcher;
using sbwt_search::QueryFileParser;
using sbwt_search::RawSequencesParser;
using sbwt_search::SdslSbwtFactory;
using std::string;

const auto kmer_size = 30;
const auto presearch_letters = 12;
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
  auto bit_seqs = sequences_parser.get_bit_seqs();
  auto positions = sequences_parser.get_positions();

  auto factory = BitVectorSbwtFactory();
  auto sbwt_parser = factory.get_sbwt_parser("path_to_sbwt_file/s");
  auto cpu_container = sbwt_parser.parse();
  auto index_builder = CpuRankIndexBuilder<
    decltype(cpu_container),
    superblock_bits,
    hyperblock_bits>(cpu_container);
  index_builder.build_index();

  auto gpu_container = cpu_container.to_gpu();
  auto presearcher = Presearcher(12, gpu_container);
  presearcher.presearch<superblock_bits, hyperblock_bits, presearch_letters>();
}
