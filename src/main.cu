#include <cmath>
#include <iostream>
#include <string>
#include <utility>

#include "ArgumentParser/ArgumentParser.hpp"
#include "Presearcher/Presearcher.cuh"
#include "QueryFileParser/QueryFileParser.h"
#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "RawSequencesParser/RawSequencesParser.hpp"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "SbwtContainer/SbwtContainer.hpp"
#include "SbwtFactory/SbwtFactory.hpp"
#include "Searcher/Searcher.cuh"
#include "Utils/CudaUtils.cuh"
#include "Utils/TypeDefinitions.h"

using sbwt_search::ACGT;
using sbwt_search::BitVectorSbwtFactory;
using sbwt_search::CharToBitsVector;
using sbwt_search::CpuRankIndexBuilder;
using sbwt_search::GpuSbwtContainer;
using sbwt_search::parse_arguments;
using sbwt_search::Presearcher;
using sbwt_search::QueryFileParser;
using sbwt_search::RawSequencesParser;
using sbwt_search::SdslSbwtFactory;
using sbwt_search::Searcher;
using sbwt_search::u64;
using std::cout;
using std::move;
using std::string;

const auto kmer_size = 30;
const auto presearch_letters = 12;
const size_t superblock_bits = 1024;
constexpr const size_t hyperblock_bits = 1ULL << 32;  // pow(2, 32);
const auto threads_per_block = 1024;

auto main(int argc, char **argv) -> int {
  /* auto args = parse_arguments(argc, argv); */
  auto query_file_parser = QueryFileParser("data/querydata/365.fna", kmer_size);
  query_file_parser.parse_kseqpp_streams();
  auto sequences_parser = RawSequencesParser<CharToBitsVector>(
    query_file_parser.get_seqs(),
    query_file_parser.get_total_positions(),
    query_file_parser.get_total_letters(),
    kmer_size
  );
  sequences_parser.parse_serial();
  vector<u64> &positions = sequences_parser.get_positions();
  vector<u64> &bit_seqs = sequences_parser.get_bit_seqs();

  auto factory = BitVectorSbwtFactory();
  auto sbwt_parser = factory.get_sbwt_parser("data/BitVectorFormat/ecoli");
  auto cpu_container = sbwt_parser.parse();
  cpu_container.change_acgt_endianness();
  auto index_builder = CpuRankIndexBuilder<
    decltype(cpu_container),
    superblock_bits,
    hyperblock_bits>(&cpu_container);
  index_builder.build_index();

  auto gpu_container = cpu_container.to_gpu();
  auto presearcher = Presearcher(gpu_container.get());
  presearcher.presearch<
    threads_per_block,
    superblock_bits,
    hyperblock_bits,
    presearch_letters>();

  auto searcher = Searcher(gpu_container.get());

  auto result = searcher.search<
    threads_per_block,
    superblock_bits,
    hyperblock_bits,
    presearch_letters,
    kmer_size>(positions, bit_seqs);

  for (int i = 0; i < 1024; ++i) { cout << result[i] << ' '; }
  cout << endl;
}
