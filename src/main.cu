#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <type_traits>

#include "ArgumentParser/ArgumentParser.hpp"
#include "FilenamesParser/FilenamesParser.h"
#include "PositionsBuilder/ContinuousPositionsBuilder.hpp"
#include "Presearcher/Presearcher.cuh"
#include "RankIndexBuilder/RankIndexBuilder.hpp"
#include "ResultsPrinter/ContinuousResultsPrinter.hpp"
#include "SbwtContainer/CpuSbwtContainer.hpp"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "SbwtContainer/SbwtContainer.hpp"
#include "SbwtFactory/SbwtFactory.hpp"
#include "Searcher/ContinuousSearcher.cuh"
#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.hpp"
#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "Utils/BenchmarkUtils.hpp"
#include "Utils/CudaUtils.cuh"
#include "Utils/MemoryUtils.hpp"
#include "Utils/TypeDefinitions.h"
#include "spdlog/cfg/env.h"
#include "spdlog/spdlog.h"

using memory_utils::get_total_system_memory;
using std::remove_reference;
using std::runtime_error;
using std::string;
using namespace sbwt_search;
using gpu_utils::get_free_gpu_memory;
using math_utils::round_down;
using std::endl;

auto get_gpu_container(string index_file) -> shared_ptr<GpuSbwtContainer>;
auto get_max_chars_per_batch(size_t unavailable_memory, uint max_batches)
  -> size_t;
auto get_max_chars_per_batch_gpu(uint max_batches) -> size_t;
auto get_max_chars_per_batch_cpu(size_t unavailable_memory, uint max_batches)
  -> size_t;

const auto program_name = "SBWT Search";
const auto program_description
  = "An application to search for k-mers in a genome given an SBWT index";
const uint kmer_size = 30;
const auto presearch_letters = 12;
const size_t superblock_bits = 1024;
constexpr const size_t hyperblock_bits = 1ULL << 32;
const auto threads_per_block = 1024;
const auto reversed_bits = true;
const auto num_seq_to_bit_converters = 3;

auto main(int argc, char **argv) -> int {
  spdlog::set_level(spdlog::level::warn);
  spdlog::cfg::load_env_levels();
  auto args = ArgumentParser(program_name, program_description, argc, argv);
  auto gpu_container = get_gpu_container(args.get_index_file());
  FilenamesParser filenames_parser(
    args.get_sequence_file(), args.get_output_file()
  );
  auto input_filenames = filenames_parser.get_input_filenames();
  auto output_filenames = filenames_parser.get_output_filenames();
  const auto max_batches = args.get_batches();
  const auto max_chars_per_batch
    = get_max_chars_per_batch(args.get_unavailable_ram(), max_batches);
  spdlog::info("Using {} characters per batch", max_chars_per_batch);
  omp_set_nested(1);
  using SequenceFileParser = ContinuousSequenceFileParser;
  auto sequence_file_parser = make_shared<SequenceFileParser>(
    input_filenames,
    kmer_size,
    max_chars_per_batch,
    max_chars_per_batch,
    num_seq_to_bit_converters,
    max_batches
  );
  using SeqToBitsConverter
    = ContinuousSeqToBitsConverter<ContinuousSequenceFileParser>;
  auto seq_to_bit_converter = make_shared<SeqToBitsConverter>(
    sequence_file_parser,
    num_seq_to_bit_converters,
    kmer_size,
    max_chars_per_batch,
    max_batches
  );
  using PositionsBuilder
    = ContinuousPositionsBuilder<ContinuousSequenceFileParser>;
  auto positions_builder = make_shared<PositionsBuilder>(
    sequence_file_parser, kmer_size, max_chars_per_batch, max_batches
  );
  using Searcher = ContinuousSearcher<
    PositionsBuilder,
    SeqToBitsConverter,
    threads_per_block,
    superblock_bits,
    hyperblock_bits,
    presearch_letters,
    reversed_bits>;
  auto searcher = make_shared<Searcher>(
    gpu_container,
    seq_to_bit_converter,
    positions_builder,
    max_batches,
    max_chars_per_batch
  );
  using ResultsPrinter = ContinuousResultsPrinter<
    Searcher,
    SequenceFileParser,
    SeqToBitsConverter>;
  auto results_printer = make_shared<ResultsPrinter>(
    searcher,
    sequence_file_parser,
    seq_to_bit_converter,
    output_filenames,
    kmer_size
  );

#pragma omp parallel sections default(shared)
  {
#pragma omp section
    sequence_file_parser->read_and_generate();
#pragma omp section
    seq_to_bit_converter->read_and_generate();
#pragma omp section
    positions_builder->read_and_generate();
#pragma omp section
    searcher->read_and_generate();
#pragma omp section
    results_printer->read_and_generate();
  }
}

auto get_gpu_container(string index_file) -> shared_ptr<GpuSbwtContainer> {
  auto sbwt_factory = SdslSbwtFactory();
  auto sbwt_parser = sbwt_factory.get_sbwt_parser(index_file);
  auto cpu_container = sbwt_parser.parse();
  using container_type = remove_reference<decltype(*cpu_container.get())>::type;
  auto index_builder
    = CpuRankIndexBuilder<container_type, superblock_bits, hyperblock_bits>(
      cpu_container
    );
  index_builder.build_index();
  auto gpu_container = cpu_container->to_gpu();
  auto presearcher = Presearcher(gpu_container);
  presearcher.presearch<
    threads_per_block,
    superblock_bits,
    hyperblock_bits,
    presearch_letters,
    reversed_bits>();
  return gpu_container;
}

auto get_max_chars_per_batch(size_t unavailable_memory, uint max_batches)
  -> size_t {
  auto gpu_chars = get_max_chars_per_batch_gpu(max_batches);
  auto cpu_chars = get_max_chars_per_batch_cpu(unavailable_memory, max_batches);
  if (gpu_chars < cpu_chars) { return gpu_chars; }
  return cpu_chars;
}

auto get_max_chars_per_batch_gpu(uint max_batches) -> size_t {
  size_t free = get_free_gpu_memory();
  auto max_chars_per_batch
    = round_down<size_t>(free * 8 / 66 / (max_batches + 1), threads_per_block);
  spdlog::debug(
    "Free gpu memory: {} bits ({:.2f}GB). This allows for {} characters per "
    "batch",
    free,
    double(free) / 8 / 1024 / 1024 / 1024,
    max_chars_per_batch
  );
  return max_chars_per_batch;
}

auto get_max_chars_per_batch_cpu(size_t unavailable_memory, uint max_batches)
  -> size_t {
  if (unavailable_memory > get_total_system_memory() * 8) {
    throw runtime_error("Not enough memory. Please specify a lower number of "
                        "unavailable-main-memory.");
  }
  size_t free = get_total_system_memory() * 8 - unavailable_memory;
  auto max_chars_per_batch
    = round_down<size_t>(free / 460 / (max_batches + 1), threads_per_block);
  spdlog::debug(
    "Free main memory: {} bits ({:.2f}GB). This allows for {} characters per "
    "batch",
    free,
    double(free) / 8 / 1024 / 1024 / 1024,
    max_chars_per_batch
  );
  return max_chars_per_batch;
}
