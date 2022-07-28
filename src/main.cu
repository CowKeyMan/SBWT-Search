#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <type_traits>

#include "ArgumentParser/ArgumentParser.hpp"
#include "Presearcher/Presearcher.cuh"
#include "QueryFileParser/QueryFileParser.h"
#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "RawSequencesParser/RawSequencesParser.hpp"
#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "SbwtContainer/SbwtContainer.hpp"
#include "SbwtFactory/SbwtFactory.hpp"
#include "Searcher/Searcher.cuh"
#include "Searcher/Searcher.hpp"
#include "Utils/BenchmarkUtils.hpp"
#include "Utils/CudaUtils.cuh"
#include "Utils/TypeDefinitions.h"

using sbwt_search::ACGT;
using sbwt_search::ArgumentParser;
using sbwt_search::BitVectorSbwtFactory;
using sbwt_search::CharToBitsVector;
using sbwt_search::CpuRankIndexBuilder;
using sbwt_search::GpuSbwtContainer;
using sbwt_search::Presearcher;
using sbwt_search::QueryFileParser;
using sbwt_search::RawSequencesParser;
using sbwt_search::SdslSbwtContainer;
using sbwt_search::SdslSbwtFactory;
using sbwt_search::SearcherCpu;
using sbwt_search::SearcherGpu;
using sbwt_search::u64;
using std::cout;
using std::move;
using std::remove_reference;
using std::shared_ptr;
using std::string;

const auto program_name = "SBWT Search";
const auto program_description
  = "An application to search for k-mers in a genome given an SBWT index";
const auto kmer_size = 30;
const auto presearch_letters = 12;
const size_t superblock_bits = 1024;
constexpr const size_t hyperblock_bits = 1ULL << 32;
const auto threads_per_block = 1024;
const auto reversed_bits = true;

auto main(int argc, char **argv) -> int {
  auto args = ArgumentParser(program_name, program_description, argc, argv);

  TIME_IT(
    auto query_file_parser = QueryFileParser(args.get_query_file(), kmer_size);
    query_file_parser.parse_kseqpp_streams();
  )
  cout << "Query File Parsing: " << TIME_IT_TOTAL << endl;
  TIME_IT(
    auto sequences_parser = RawSequencesParser<CharToBitsVector>(
      query_file_parser.get_seqs(),
      query_file_parser.get_total_positions(),
      query_file_parser.get_total_letters(),
      kmer_size
    );
    sequences_parser.parse_serial();
  )
  cout << "Sequence Parsing: " << TIME_IT_TOTAL << endl;
  shared_ptr<vector<u64>> positions = sequences_parser.get_positions();
  shared_ptr<vector<u64>> bit_seqs = sequences_parser.get_bit_seqs();

  auto factory = SdslSbwtFactory();
  TIME_IT(
    auto sbwt_parser = factory.get_sbwt_parser(args.get_index_file());
    auto cpu_container = sbwt_parser.parse();
  )
  cout << "SBWT parsing: " << TIME_IT_TOTAL << endl;
  using container_type = remove_reference<decltype(*cpu_container.get())>::type;
  TIME_IT(
    auto index_builder = CpuRankIndexBuilder<container_type, superblock_bits, hyperblock_bits>(cpu_container);
    index_builder.build_index();
  )
  cout << "Index Building: " << TIME_IT_TOTAL << endl;
  TIME_IT(
    auto gpu_container = cpu_container->to_gpu();
  )
  cout << "Copying to gpu: " << TIME_IT_TOTAL << endl;
  TIME_IT(
    auto presearcher = Presearcher(gpu_container);
    presearcher.presearch<
      threads_per_block,
      superblock_bits,
      hyperblock_bits,
      presearch_letters,
      reversed_bits>();
  )
  cout << "Presearching: " << TIME_IT_TOTAL << endl;
  TIME_IT(
    auto searcher = SearcherGpu(gpu_container);
    auto result = searcher.search<
      threads_per_block,
      superblock_bits,
      hyperblock_bits,
      presearch_letters,
      kmer_size,
      reversed_bits>(*positions.get(), *bit_seqs.get());
  )
  cout << "Searching (+ copying positions to gpu and gettings results back): " << TIME_IT_TOTAL << endl;
  for (auto x: result) { cout << x << ' '; }
  cout << endl;
}
