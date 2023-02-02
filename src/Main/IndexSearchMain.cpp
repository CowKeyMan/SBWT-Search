#include <omp.h>
#include <stdexcept>
#include <string>

#include "ArgumentParser/IndexSearchArgumentParser.h"
#include "FilenamesParser/FilenamesParser.h"
#include "Global/GlobalDefinitions.h"
#include "Main/IndexSearchMain.h"
#include "Presearcher/Presearcher.h"
#include "SbwtBuilder/SbwtBuilder.h"
#include "SbwtContainer/CpuSbwtContainer.h"
#include "SbwtContainer/GpuSbwtContainer.h"
#include "Tools/GpuUtils.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/MemoryUtils.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using gpu_utils::get_free_gpu_memory;
using log_utils::Logger;
using math_utils::round_down;
using memory_utils::get_total_system_memory;
using std::min;
using std::runtime_error;
using std::to_string;

auto IndexSearchMain::main(int argc, char **argv) -> int {
  const string program_name = "SBWT_Search";
  const string program_description
    = "An application to search for k-mers in a genome given an SBWT index";
  Logger::initialise_global_logging(Logger::LOG_LEVEL::WARN);
  Logger::log_timed_event("main", Logger::EVENT_STATE::START);
  auto args
    = IndexSearchArgumentParser(program_name, program_description, argc, argv);
  Logger::log(Logger::LOG_LEVEL::INFO, "Loading components into memory");
  auto gpu_container = get_gpu_container(args.get_index_file());
  kmer_size = gpu_container->get_kmer_size();
  load_input_output_filenames(args.get_sequence_file(), args.get_output_file());
  load_batch_info(
    args.get_batches(), args.get_unavailable_ram(), args.get_max_cpu_memory()
  );
  load_threads();
  auto
    [sequence_file_parser,
     seq_to_bits_converter,
     positions_builder,
     searcher,
     results_printer]
    = get_components(gpu_container, args.get_print_mode());
  Logger::log(Logger::LOG_LEVEL::INFO, "Running queries");
  run_components(
    sequence_file_parser,
    seq_to_bits_converter,
    positions_builder,
    searcher,
    results_printer
  );
  Logger::log(Logger::LOG_LEVEL::INFO, "Finished");
  Logger::log_timed_event("main", Logger::EVENT_STATE::STOP);
  return 0;
}

auto IndexSearchMain::get_gpu_container(const string &index_file)
  -> shared_ptr<GpuSbwtContainer> {
  Logger::log_timed_event("SBWTLoader", Logger::EVENT_STATE::START);
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
  Logger::log_timed_event("SBWTLoader", Logger::EVENT_STATE::STOP);
  return gpu_container;
}

auto IndexSearchMain::load_input_output_filenames(
  const string &input_file, const string &output_file
) -> void {
  FilenamesParser filenames_parser(input_file, output_file);
  input_filenames = filenames_parser.get_input_filenames();
  output_filenames = filenames_parser.get_output_filenames();
  if (input_filenames.size() != output_filenames.size()) {
    throw runtime_error("Input and output file sizes differ");
  }
}

auto IndexSearchMain::load_batch_info(
  u64 max_batches_, u64 unavailable_ram, u64 max_cpu_memory
) -> void {
  max_batches = max_batches_;
  max_chars_per_batch
    = get_max_chars_per_batch(unavailable_ram, max_cpu_memory);
  if (max_chars_per_batch == 0) { throw runtime_error("Not enough memory"); }
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    "Using " + to_string(max_chars_per_batch) + " characters per batch"
  );
}

auto IndexSearchMain::get_max_chars_per_batch(
  size_t unavailable_memory, size_t max_cpu_memory
) -> size_t {
  auto gpu_chars = get_max_chars_per_batch_gpu();
  auto cpu_chars
    = get_max_chars_per_batch_cpu(unavailable_memory, max_cpu_memory);
  return min(gpu_chars, cpu_chars);
}

auto IndexSearchMain::get_max_chars_per_batch_gpu() -> size_t {
  size_t free = get_free_gpu_memory() * 8 * 0.95;
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

auto IndexSearchMain::get_max_chars_per_batch_cpu(
  size_t unavailable_memory, size_t max_memory
) -> size_t {
  if (unavailable_memory > get_total_system_memory() * sizeof(size_t)) {
    throw runtime_error("Not enough memory. Please specify a lower number of "
                        "unavailable-main-memory.");
  }
  size_t free
    = (get_total_system_memory() * sizeof(size_t) - unavailable_memory) * 0.95;
  if (max_memory < free) { free = max_memory; }
  auto max_chars_per_batch
    = round_down<size_t>(free / 146 / max_batches, threads_per_block);
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
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

auto IndexSearchMain::load_threads() -> void {
  omp_set_nested(1);
#pragma omp parallel
#pragma omp single
  threads = omp_get_num_threads();
  Logger::log(
    Logger::LOG_LEVEL::INFO, format("Running OpenMP with {} threads", threads)
  );
}

auto IndexSearchMain::get_components(
  const shared_ptr<GpuSbwtContainer> &gpu_container, const string &print_mode
)
  -> std::tuple<
    shared_ptr<ContinuousSequenceFileParser>,
    shared_ptr<ContinuousSeqToBitsConverter>,
    shared_ptr<ContinuousPositionsBuilder>,
    shared_ptr<ContinuousSearcher>,
    ResultsPrinter> {
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::START);
  auto sequence_file_parser = make_shared<ContinuousSequenceFileParser>(
    input_filenames, kmer_size, max_chars_per_batch, max_batches
  );

  auto seq_to_bits_converter = make_shared<ContinuousSeqToBitsConverter>(
    sequence_file_parser->get_string_sequence_batch_producer(),
    threads,
    kmer_size,
    max_chars_per_batch,
    max_batches
  );

  auto positions_builder = make_shared<ContinuousPositionsBuilder>(
    sequence_file_parser->get_string_break_batch_producer(),
    kmer_size,
    max_chars_per_batch,
    max_batches
  );

  auto searcher = make_shared<ContinuousSearcher>(
    gpu_container,
    seq_to_bits_converter->get_bits_producer(),
    positions_builder,
    max_batches,
    max_chars_per_batch
  );

  ResultsPrinter results_printer = get_results_printer(
    print_mode,
    searcher,
    sequence_file_parser->get_interval_batch_producer(),
    seq_to_bits_converter->get_invalid_chars_producer()
  );
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::STOP);

  return {
    std::move(sequence_file_parser),
    std::move(seq_to_bits_converter),
    std::move(positions_builder),
    std::move(searcher),
    std::move(results_printer)};
}

auto IndexSearchMain::get_results_printer(
  const string &print_mode,
  const shared_ptr<ContinuousSearcher> &searcher,
  const shared_ptr<IntervalBatchProducer> &interval_batch_producer,
  const shared_ptr<InvalidCharsProducer> &invalid_chars_producer
) -> ResultsPrinter {
  if (print_mode == "ascii") {
    return make_shared<AsciiContinuousResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size
    );
  }
  if (print_mode == "binary") {
    return make_shared<BinaryContinuousResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size
    );
  }
  if (print_mode == "bool") {
    return make_shared<BoolContinuousResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size,
      max_chars_per_batch
    );
  }
  throw runtime_error("Invalid value passed by user for print_mode");
}

auto IndexSearchMain::run_components(
  shared_ptr<ContinuousSequenceFileParser> &sequence_file_parser,
  shared_ptr<ContinuousSeqToBitsConverter> &seq_to_bits_converter,
  shared_ptr<ContinuousPositionsBuilder> &positions_builder,
  shared_ptr<ContinuousSearcher> &searcher,
  ResultsPrinter &results_printer
) -> void {
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::START);
#pragma omp parallel sections num_threads(5)
  {
#pragma omp section
    { sequence_file_parser->read_and_generate(); }
#pragma omp section
    { seq_to_bits_converter->read_and_generate(); }
#pragma omp section
    { positions_builder->read_and_generate(); }
#pragma omp section
    { searcher->read_and_generate(); }
#pragma omp section
    {
      std::visit(
        [](auto &arg) -> void { arg->read_and_generate(); }, results_printer
      );
    }
  }
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::STOP);
}

}  // namespace sbwt_search
