#include <iostream>
#include <string>

#include <unordered_set>

#include "Utils/BenchmarkUtils.hpp"
#include "Utils/TypeDefinitions.h"
#include "cxxopts.hpp"

using std::cerr;
using std::cout;
using std::string;
using std::unordered_set;

using namespace sbwt_search;

auto parse_arguments(int argc, char **argv) -> unordered_set<string>;

u64 kmer_size = 30;

const auto input_file_paths
  = { "benchmark_objects/FASTA1GB.fna", "benchmark_objects/fastq1GB.fnq" };

auto raw_sequence_parser_char_to_bits() -> void;

auto main(int argc, char **argv) -> int {
  auto arguments = parse_arguments(argc, argv);
}

auto parse_arguments(int argc, char **argv) -> unordered_set<string> {
  auto options = cxxopts::Options("Benchmark");
  options.allow_unrecognised_options();
  auto local_arguments = options.parse(argc, argv);
  auto unmatched_args = local_arguments.unmatched();
  return unordered_set<string>(
    std::make_move_iterator(unmatched_args.begin()),
    std::make_move_iterator(unmatched_args.end())
  );
}
