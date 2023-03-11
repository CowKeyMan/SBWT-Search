#include <cstdlib>
#include <iostream>
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
#include "Tools/StdUtils.hpp"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using gpu_utils::get_free_gpu_memory;
using log_utils::Logger;
using math_utils::bits_to_gB;
using math_utils::round_down;
using memory_utils::get_total_system_memory;
using std::cerr;
using std::endl;
using std::min;
using std::runtime_error;
using std::to_string;
using std_utils::split_vector;

const u64 num_components = 5;

auto IndexSearchMain::main(int argc, char **argv) -> int {
  const string program_name = "index";
  const string program_description = "sbwt_search";
  Logger::log_timed_event("main", Logger::EVENT_STATE::START);
  args = make_unique<IndexSearchArgumentParser>(
    program_name, program_description, argc, argv
  );
  Logger::log(Logger::LOG_LEVEL::INFO, "Loading components into memory");
  auto gpu_container = get_gpu_container();
  kmer_size = gpu_container->get_kmer_size();
  auto [split_input_filenames, split_output_filenames]
    = get_input_output_filenames();
  load_batch_info();
  omp_set_nested(1);
  load_threads();
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    format("Running OpenMP with {} threads", get_threads())
  );
  auto
    [sequence_file_parsers,
     seq_to_bits_converters,
     positions_builders,
     searchers,
     results_printers]
    = get_components(
      gpu_container, split_input_filenames, split_output_filenames
    );
  Logger::log(Logger::LOG_LEVEL::INFO, "Running queries");
  run_components(
    sequence_file_parsers,
    seq_to_bits_converters,
    positions_builders,
    searchers,
    results_printers
  );
  Logger::log(Logger::LOG_LEVEL::INFO, "Finished");
  Logger::log_timed_event("main", Logger::EVENT_STATE::STOP);
  return 0;
}

auto IndexSearchMain::get_args() const -> const IndexSearchArgumentParser & {
  return *args;
}

auto IndexSearchMain::get_gpu_container() -> shared_ptr<GpuSbwtContainer> {
  Logger::log_timed_event("SBWTLoader", Logger::EVENT_STATE::START);
  Logger::log_timed_event("SBWTParserAndIndex", Logger::EVENT_STATE::START);
  auto builder = SbwtBuilder(get_args().get_index_file());
  auto cpu_container = builder.get_cpu_sbwt();
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

auto IndexSearchMain::load_batch_info() -> void {
  max_chars_per_batch = get_max_chars_per_batch();
  max_reads_per_batch
    = max_chars_per_batch / get_args().get_base_pairs_per_read();
  if (max_chars_per_batch == 0) { throw runtime_error("Not enough memory"); }
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    format(
      "Using {} max characters per batch and {} max reads per batch",
      max_chars_per_batch,
      max_reads_per_batch
    )
  );
}

auto IndexSearchMain::get_max_chars_per_batch() -> u64 {
  if (batches == 0) {
    cerr << "Initialise batches before max_chars_per_batch" << endl;
    std::quick_exit(1);
  }
  auto gpu_chars = get_max_chars_per_batch_gpu();
  auto cpu_chars = get_max_chars_per_batch_cpu();
  return round_down<u64>(
    min(gpu_chars, cpu_chars) / batches, threads_per_block
  );
}

auto IndexSearchMain::get_max_chars_per_batch_gpu() -> u64 {
  u64 free = static_cast<u64>(
    static_cast<double>(get_free_gpu_memory() * sizeof(u64))
    * get_args().get_gpu_memory_percentage()
  );
  // 64 for each position where each result is also stored
  // 2 for each base pair since these are bit packed
  const u64 bits_required_per_character = 64 + 2;
  auto max_chars_per_batch = free / bits_required_per_character;
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Free gpu memory: {} bits ({:.2f}GB). This allows for {} characters per "
      "batch",
      free,
      bits_to_gB(free),
      max_chars_per_batch
    )
  );
  return max_chars_per_batch;
}

auto IndexSearchMain::get_max_chars_per_batch_cpu() -> u64 {
  if (get_args().get_unavailable_ram() > get_total_system_memory() * sizeof(u64)) {
    throw runtime_error("Not enough memory. Please specify a lower number of "
                        "unavailable-main-memory.");
  }
  u64 free_bits = static_cast<u64>(
    static_cast<double>(
      min(
        get_total_system_memory() * sizeof(u64), get_args().get_max_cpu_memory()
      )
      - get_args().get_unavailable_ram()
    )
    * get_args().get_cpu_memory_percentage()
  );
  // 8 bits per character for string sequence batch when reading
  // 8 bits per character for invalid characters
  // 64 bits for the positions
  // 64 bits for the results
  // 2 bits for the bit packed sequences
  // 20 * 8 bits for each printed character
  // 1 * 8 bits for each space after the printed characters
  // 64 bits per read for the chars_before_newline
  // 8 bits per read for each newline
  const u64 bits_required_per_character = static_cast<u64>(
    static_cast<double>(8 + 64 + 64 + 8 + 2 + (20 + 1) * 8)
    + (1.0 / static_cast<double>(get_args().get_base_pairs_per_read()))
      * (64.0 + 8.0)
  );
  auto max_chars_per_batch
    = free_bits / bits_required_per_character / num_components;
  Logger::log(
    Logger::LOG_LEVEL::DEBUG,
    format(
      "Free main memory: {} bits ({:.2f}GB). This allows for {} "
      "characters per batch",
      free_bits,
      bits_to_gB(free_bits),
      max_chars_per_batch
    )
  );
  return max_chars_per_batch;
}

auto IndexSearchMain::get_components(
  const shared_ptr<GpuSbwtContainer> &gpu_container,
  const vector<vector<string>> &split_input_filenames,
  const vector<vector<string>> &split_output_filenames
)
  -> std::tuple<
    vector<shared_ptr<ContinuousSequenceFileParser>>,
    vector<shared_ptr<ContinuousSeqToBitsConverter>>,
    vector<shared_ptr<ContinuousPositionsBuilder>>,
    vector<shared_ptr<ContinuousSearcher>>,
    vector<ResultsPrinter>> {
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::START);
  vector<shared_ptr<ContinuousSequenceFileParser>> sequence_file_parsers;
  vector<shared_ptr<ContinuousSeqToBitsConverter>> seq_to_bits_converters;
  vector<shared_ptr<ContinuousPositionsBuilder>> positions_builders;
  vector<shared_ptr<ContinuousSearcher>> searchers;
  vector<ResultsPrinter> results_printers;
  for (u64 i = 0; i < split_input_filenames.size(); ++i) {
    sequence_file_parsers.push_back(make_shared<ContinuousSequenceFileParser>(
      split_input_filenames[i],
      kmer_size,
      max_chars_per_batch,
      max_reads_per_batch,
      num_components
    ));

    seq_to_bits_converters.push_back(make_shared<ContinuousSeqToBitsConverter>(
      sequence_file_parsers[i]->get_string_sequence_batch_producer(),
      get_threads(),
      kmer_size,
      max_chars_per_batch,
      num_components
    ));

    positions_builders.push_back(make_shared<ContinuousPositionsBuilder>(
      sequence_file_parsers[i]->get_string_break_batch_producer(),
      kmer_size,
      max_chars_per_batch,
      num_components
    ));

    searchers.push_back(make_shared<ContinuousSearcher>(
      gpu_container,
      seq_to_bits_converters[i]->get_bits_producer(),
      positions_builders[i],
      num_components,
      max_chars_per_batch
    ));

    results_printers.push_back(get_results_printer(
      searchers[i],
      sequence_file_parsers[i]->get_interval_batch_producer(),
      seq_to_bits_converters[i]->get_invalid_chars_producer(),
      split_output_filenames[i]
    ));
  }
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::STOP);

  return {
    std::move(sequence_file_parsers),
    std::move(seq_to_bits_converters),
    std::move(positions_builders),
    std::move(searchers),
    std::move(results_printers)};
}

auto IndexSearchMain::get_results_printer(
  const shared_ptr<ContinuousSearcher> &searcher,
  const shared_ptr<IntervalBatchProducer> &interval_batch_producer,
  const shared_ptr<InvalidCharsProducer> &invalid_chars_producer,
  const vector<string> &output_filenames
) -> ResultsPrinter {
  if (get_args().get_print_mode() == "ascii") {
    return make_shared<AsciiContinuousResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size,
      get_threads(),
      max_chars_per_batch
    );
  }
  if (get_args().get_print_mode() == "binary") {
    return make_shared<BinaryContinuousResultsPrinter>(
      searcher,
      interval_batch_producer,
      invalid_chars_producer,
      output_filenames,
      kmer_size,
      get_threads(),
      max_chars_per_batch
    );
  }
  throw runtime_error("Invalid value passed by user for print_mode");
}

auto IndexSearchMain::get_input_output_filenames()
  -> std::tuple<vector<vector<string>>, vector<vector<string>>> {
  FilenamesParser filenames_parser(
    get_args().get_query_file(), get_args().get_output_file()
  );
  auto all_input_filenames = filenames_parser.get_input_filenames();
  auto all_output_filenames = filenames_parser.get_output_filenames();
  if (all_input_filenames.size() != all_output_filenames.size()) {
    throw runtime_error("Input and output file sizes differ");
  }
  batches = std::min(args->get_batches(), all_input_filenames.size());
  return {
    split_vector(all_input_filenames, batches),
    split_vector(all_output_filenames, batches)};
}

auto IndexSearchMain::run_components(
  vector<shared_ptr<ContinuousSequenceFileParser>> &sequence_file_parsers,
  vector<shared_ptr<ContinuousSeqToBitsConverter>> &seq_to_bits_converters,
  vector<shared_ptr<ContinuousPositionsBuilder>> &positions_builders,
  vector<shared_ptr<ContinuousSearcher>> &searchers,
  vector<ResultsPrinter> &results_printers
) -> void {
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::START);
#pragma omp parallel sections num_threads(num_components)
  {
#pragma omp section
    {
#pragma omp parallel for num_threads(batches)
      for (auto &element : sequence_file_parsers) {
        element->read_and_generate();
      }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(batches)
      for (auto &element : seq_to_bits_converters) {
        element->read_and_generate();
      }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(batches)
      for (auto &element : positions_builders) { element->read_and_generate(); }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(batches)
      for (auto &element : searchers) { element->read_and_generate(); }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(batches)
      for (auto &element : results_printers) {
        std::visit(
          [](auto &arg) -> void { arg->read_and_generate(); }, element
        );
      }
    }
  }
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::STOP);
}

}  // namespace sbwt_search
