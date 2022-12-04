#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <type_traits>

#include "ArgumentParser/ArgumentParser.hpp"
#include "FilenamesParser/FilenamesParser.h"
#include "PoppyBuilder/PoppyBuilder.h"
#include "PositionsBuilder/ContinuousPositionsBuilder.hpp"
#include "Presearcher/Presearcher.cuh"
#include "ResultsPrinter/AsciiContinuousResultsPrinter.hpp"
#include "ResultsPrinter/BinaryContinuousResultsPrinter.hpp"
#include "ResultsPrinter/BoolContinuousResultsPrinter.hpp"
#include "SbwtBuilder/SbwtBuilder.h"
#include "SbwtContainer/GpuSbwtContainer.cuh"
#include "SbwtContainer/SbwtContainer.h"
#include "Searcher/ContinuousSearcher.cuh"
#include "SeqToBitsConverter/BitsProducer.hpp"
#include "SeqToBitsConverter/ContinuousSeqToBitsConverter.hpp"
#include "SeqToBitsConverter/InvalidCharsProducer.hpp"
#include "SequenceFileParser/ContinuousSequenceFileParser.h"
#include "SequenceFileParser/IntervalBatchProducer.h"
#include "SequenceFileParser/StringBreakBatchProducer.h"
#include "SequenceFileParser/StringSequenceBatchProducer.h"
#include "Utils/BenchmarkUtils.hpp"
#include "Utils/CudaUtils.cuh"
#include "Utils/Logger.h"
#include "Utils/MemoryUtils.hpp"
#include "Utils/TypeDefinitions.h"
#include "fmt/core.h"

using memory_utils::get_total_system_memory;
using std::remove_reference;
using std::runtime_error;
using std::string;
using namespace sbwt_search;
using fmt::format;
using gpu_utils::get_free_gpu_memory;
using log_utils::Logger;
using math_utils::round_down;
using std::endl;
constexpr auto WARN = Logger::LOG_LEVEL::WARN;
constexpr auto INFO = Logger::LOG_LEVEL::INFO;
constexpr auto DEBUG = Logger::LOG_LEVEL::DEBUG;

auto get_gpu_container(string index_file) -> shared_ptr<GpuSbwtContainer>;
auto get_max_chars_per_batch(
  size_t unavailable_memory, uint max_batches, size_t max_cpu_memory
) -> size_t;
auto get_max_chars_per_batch_gpu(uint max_batches) -> size_t;
auto get_max_chars_per_batch_cpu(
  size_t unavailable_memory, uint max_batches, size_t max_memory
) -> size_t;

const auto program_name = "SBWT Search";
const auto program_description
  = "An application to search for k-mers in a genome given an SBWT index";

auto main(int argc, char **argv) -> int {
  Logger::initialise_global_logging(WARN);
  Logger::log_timed_event("main", Logger::EVENT_STATE::START);
  Logger::log_timed_event("SBWTLoader", Logger::EVENT_STATE::START);
  auto args = ArgumentParser(program_name, program_description, argc, argv);
  auto gpu_container = get_gpu_container(args.get_index_file());
  const uint kmer_size = gpu_container->get_kmer_size();
  Logger::log_timed_event("SBWTLoader", Logger::EVENT_STATE::STOP);
  FilenamesParser filenames_parser(
    args.get_sequence_file(), args.get_output_file()
  );
  auto input_filenames = filenames_parser.get_input_filenames();
  auto output_filenames = filenames_parser.get_output_filenames();
  if (input_filenames != output_filenames) {
    runtime_error("Input and output file sizes differ");
  }
  const auto max_batches = args.get_batches();
  const auto max_chars_per_batch = get_max_chars_per_batch(
    args.get_unavailable_ram(), max_batches, args.get_max_cpu_memory()
  );
  if (max_chars_per_batch == 0) { throw runtime_error("Not enough memory"); }
  Logger::log(
    INFO, "Using " + to_string(max_chars_per_batch) + " characters per batch"
  );
  omp_set_nested(1);
  uint threads = 0;
#pragma omp parallel
#pragma omp single
  threads = omp_get_num_threads();
  Logger::log(INFO, format("Running OpenMP with {} threads", threads));
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::START);
  using SequenceFileParser = ContinuousSequenceFileParser;
  auto string_sequence_batch_producer
    = make_shared<StringSequenceBatchProducer>(max_batches);
  auto string_break_batch_producer
    = make_shared<StringBreakBatchProducer>(max_batches);
  auto interval_batch_producer
    = make_shared<IntervalBatchProducer>(max_batches);
  auto sequence_file_parser = make_shared<SequenceFileParser>(
    input_filenames,
    kmer_size,
    max_chars_per_batch,
    max_batches,
    string_sequence_batch_producer,
    string_break_batch_producer,
    interval_batch_producer
  );

  using SeqToBitsConverter
    = ContinuousSeqToBitsConverter<StringSequenceBatchProducer>;
  using InvalidCharsProducer
    = InvalidCharsProducer<StringSequenceBatchProducer>;
  using BitsProducer = BitsProducer<StringSequenceBatchProducer>;
  auto invalid_chars_producer = make_shared<InvalidCharsProducer>(
    kmer_size, max_chars_per_batch, max_batches
  );
  auto bits_producer
    = make_shared<BitsProducer>(max_chars_per_batch, max_batches);
  auto seq_to_bit_converter = make_shared<SeqToBitsConverter>(
    string_sequence_batch_producer,
    invalid_chars_producer,
    bits_producer,
    threads
  );

  using PositionsBuilder = ContinuousPositionsBuilder<StringBreakBatchProducer>;
  auto positions_builder = make_shared<PositionsBuilder>(
    string_break_batch_producer, kmer_size, max_chars_per_batch, max_batches
  );

  using Searcher = ContinuousSearcher<PositionsBuilder, BitsProducer>;
  auto searcher = make_shared<Searcher>(
    gpu_container,
    bits_producer,
    positions_builder,
    max_batches,
    max_chars_per_batch
  );

  using ResultsPrinter = ContinuousResultsPrinter<
    Searcher,
    IntervalBatchProducer,
    InvalidCharsProducer>;
  using AsciiResultsPrinter = AsciiContinuousResultsPrinter<
    Searcher,
    IntervalBatchProducer,
    InvalidCharsProducer>;
  using BinaryResultsPrinter = BinaryContinuousResultsPrinter<
    Searcher,
    IntervalBatchProducer,
    InvalidCharsProducer>;
  using BoolResultsPrinter = BoolContinuousResultsPrinter<
    Searcher,
    IntervalBatchProducer,
    InvalidCharsProducer>;
  shared_ptr<ResultsPrinter> results_printer;
  if (args.get_print_mode() == "ascii") {
    results_printer = make_shared<AsciiResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size
    );
  } else if (args.get_print_mode() == "binary") {
    results_printer = make_shared<BinaryResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size
    );
  } else if (args.get_print_mode() == "bool") {
    results_printer = make_shared<BoolResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size,
      max_chars_per_batch
    );
  } else {
    throw runtime_error("Invalid value passed by user for print_mode");
  }
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::STOP);
#pragma omp parallel sections default(shared)
  {
#pragma omp section
    { sequence_file_parser->read_and_generate(); }
#pragma omp section
    { seq_to_bit_converter->read_and_generate(); }
#pragma omp section
    { positions_builder->read_and_generate(); }
#pragma omp section
    { searcher->read_and_generate(); }
#pragma omp section
    { results_printer->read_and_generate(); }
  }
  Logger::log(INFO, "DONE");
  Logger::log_timed_event("main", Logger::EVENT_STATE::STOP);
}

auto get_gpu_container(string index_file) -> shared_ptr<GpuSbwtContainer> {
  Logger::log_timed_event("SBWTParserAndIndex", Logger::EVENT_STATE::START);
  auto builder = SbwtBuilder(index_file);
  auto cpu_container = builder.get_cpu_sbwt(true);
  Logger::log_timed_event("SBWTParserAndIndex", Logger::EVENT_STATE::STOP);
  Logger::log_timed_event("SBWT_GPU_Transfer", Logger::EVENT_STATE::START);
  auto gpu_container = cpu_container->to_gpu();
  Logger::log_timed_event("SBWT_GPU_Transfer", Logger::EVENT_STATE::STOP);
  auto presearcher = Presearcher(gpu_container);
  Logger::log_timed_event("Presearcher", Logger::EVENT_STATE::START);
  presearcher.presearch();
  Logger::log_timed_event("Presearcher", Logger::EVENT_STATE::STOP);
  return gpu_container;
}

auto get_max_chars_per_batch(
  size_t unavailable_memory, uint max_batches, size_t max_cpu_memory
) -> size_t {
  auto gpu_chars = get_max_chars_per_batch_gpu(max_batches);
  auto cpu_chars = get_max_chars_per_batch_cpu(
    unavailable_memory, max_batches, max_cpu_memory
  );
  if (gpu_chars < cpu_chars) { return gpu_chars; }
  return cpu_chars;
}

auto get_max_chars_per_batch_gpu(uint max_batches) -> size_t {
  size_t free = get_free_gpu_memory() * 8;
  auto max_chars_per_batch = round_down<size_t>(free / 66, threads_per_block);
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Free gpu memory: {} bits ({:.2f}GB). This allows for {} characters per "
      "batch",
      free,
      double(free) / 8 / 1024 / 1024 / 1024,
      max_chars_per_batch
    )
  );
  return max_chars_per_batch;
}

auto get_max_chars_per_batch_cpu(
  size_t unavailable_memory, uint max_batches, size_t max_memory
) -> size_t {
  if (unavailable_memory > get_total_system_memory() * 8) {
    throw runtime_error("Not enough memory. Please specify a lower number of "
                        "unavailable-main-memory.");
  }
  size_t free = get_total_system_memory() * 8 - unavailable_memory;
  if (max_memory < free) { free = max_memory; }
  auto max_chars_per_batch
    = round_down<size_t>(free / (146 * 1.06) / max_batches, threads_per_block);
  Logger::log(
    DEBUG,
    format(
      "Free main memory: {} bits ({:.2f}GB). This allows for {} "
      "characters per batch",
      free,
      double(free) / 8 / 1024 / 1024 / 1024,
      max_chars_per_batch
    )
  );
  return max_chars_per_batch;
}
