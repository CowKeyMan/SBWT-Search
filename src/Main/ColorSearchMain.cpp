#include <iostream>
#include <memory>
#include <omp.h>
#include <stdexcept>
#include <string>

#include "ArgumentParser/ColorSearchArgumentParser.h"
#include "ColorIndexBuilder/ColorIndexBuilder.h"
#include "FilenamesParser/FilenamesParser.h"
#include "FilesizeLoadBalancer/FilesizeLoadBalancer.h"
#include "Global/GlobalDefinitions.h"
#include "Main/ColorSearchMain.h"
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
using math_utils::divide_and_ceil;
using math_utils::divide_and_round;
using math_utils::round_down;
using memory_utils::get_total_system_memory;
using std::cerr;
using std::endl;
using std::make_shared;
using std::min;
using std::runtime_error;

const u64 num_components = 4;

auto ColorSearchMain::main(int argc, char **argv) -> int {
  const string program_name = "colors";
  const string program_description = "sbwt_search";
  Logger::log_timed_event("main", Logger::EVENT_STATE::START);
  args = make_unique<ColorSearchArgumentParser>(
    program_name, program_description, argc, argv
  );
  load_threads();
  Logger::log(Logger::LOG_LEVEL::INFO, "Loading components into memory");
  auto gpu_container = get_gpu_container();
  num_colors = gpu_container->num_colors;
  auto [input_filenames, output_filenames] = get_input_output_filenames();
  load_batch_info();
  omp_set_nested(1);
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    format("Running OpenMP with {} threads", get_threads())
  );
  auto [index_file_parser, searcher, post_processor, results_printer]
    = get_components(gpu_container, input_filenames, output_filenames);
  Logger::log(Logger::LOG_LEVEL::INFO, "Running queries");
  run_components(index_file_parser, searcher, post_processor, results_printer);
  Logger::log(Logger::LOG_LEVEL::INFO, "Finished");
  Logger::log_timed_event("main", Logger::EVENT_STATE::STOP);
  return 0;
}

inline auto ColorSearchMain::get_gpu_container()
  -> shared_ptr<GpuColorIndexContainer> {
  Logger::log_timed_event("ColorsLoader", Logger::EVENT_STATE::START);
  auto color_index_builder = ColorIndexBuilder(get_args().get_colors_file());
  auto cpu_container = color_index_builder.get_cpu_color_index_container();
  auto gpu_container = cpu_container.to_gpu();
  Logger::log_timed_event("ColorsLoader", Logger::EVENT_STATE::STOP);
  return gpu_container;
}

auto ColorSearchMain::load_batch_info() -> void {
  max_indexes_per_batch = get_max_chars_per_batch();
  max_reads_per_batch
    = max_indexes_per_batch / get_args().get_indexes_per_read();
  if (max_indexes_per_batch == 0) { throw runtime_error("Not enough memory"); }
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    format(
      "Using {} max indexes per batch and {} max reads per batch",
      max_indexes_per_batch,
      max_reads_per_batch
    )
  );
}

auto ColorSearchMain::get_max_chars_per_batch() -> u64 {
  if (streams == 0) {
    cerr << "ERROR: Initialise batches before max_chars_per_batch" << endl;
    std::quick_exit(1);
  }
  auto cpu_chars = get_max_chars_per_batch_cpu();
#if defined(__HIP_CPU_RT__)
  auto gpu_chars = numeric_limits<u64>::max();
#else
  auto gpu_chars = get_max_chars_per_batch_gpu();
#endif
  return round_down<u64>(min(cpu_chars, gpu_chars), threads_per_block);
}

auto ColorSearchMain::get_max_chars_per_batch_gpu() -> u64 {
  u64 free = static_cast<u64>(
    static_cast<double>(get_free_gpu_memory() * bits_in_byte)
    * get_args().get_gpu_memory_percentage()
  );
  // 64 for each index found
  // The results take: num_colors * 64 / warp_size
  const u64 bits_required_per_character = static_cast<u64>(
    64 + divide_and_ceil<u64>(64 * num_colors, gpu_warp_size)
  );
  auto max_chars_per_batch = free / bits_required_per_character / streams;
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

auto ColorSearchMain::get_max_chars_per_batch_cpu() -> u64 {
  if (get_args().get_unavailable_ram() > get_total_system_memory() * bits_in_byte) {
    throw runtime_error("Not enough memory. Please specify a lower number of "
                        "unavailable-main-memory.");
  }
  u64 available_ram = min(
    get_total_system_memory() * bits_in_byte, get_args().get_max_cpu_memory()
  );
  u64 unavailable_ram = get_args().get_unavailable_ram();
  u64 free_bits = (unavailable_ram > available_ram) ?
    0 :
    static_cast<u64>(
      static_cast<double>(available_ram - unavailable_ram)
      * get_args().get_cpu_memory_percentage()
    );
  u64 bits_reserved_for_results_printer = divide_and_round<u64>(free_bits, 3);
  results_printer_max_reads_in_buffer = bits_reserved_for_results_printer
    / get_threads() / streams
    / (num_colors * (max_chars_in_u64 + 1) * bits_in_byte + bits_in_byte);
  // 64 bits per index
  // 64 * num_colors / warp_size to store the results
  // 20 * 8 bits for each printed result
  // 1 * 8 bits for each space after the printed result
  // 8 bits for each printed newline
  // 64 bits per read for each newline
  // 64 bits per read for each found result count
  // 64 bits per read for each not found result count
  // 64 bits per read for each invalid result count
  const u64 bits_required_per_character = static_cast<u64>(
    64 + divide_and_ceil<u64>(64 * num_colors, gpu_warp_size)
    + divide_and_ceil<u64>(64 * 4 + 8, get_args().get_indexes_per_read())
#if defined(__HIP_CPU_RT__)  // include gpu required memory as well
    + static_cast<u64>(
      64.0
      + static_cast<double>(num_colors) * 64.0
        / static_cast<double>(gpu_warp_size)
    )
#endif
  );
  free_bits -= bits_reserved_for_results_printer;
  auto max_chars_per_batch
    = free_bits / bits_required_per_character / num_components / streams;
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

auto ColorSearchMain::get_input_output_filenames()
  -> std::tuple<vector<vector<string>>, vector<vector<string>>> {
  FilenamesParser filenames_parser(
    get_args().get_query_file(), get_args().get_output_file()
  );
  auto input_filenames = filenames_parser.get_input_filenames();
  auto output_filenames = filenames_parser.get_output_filenames();
  if (input_filenames.size() != output_filenames.size()) {
    throw runtime_error("Input and output file sizes differ");
  }
  streams = min(input_filenames.size(), args->get_streams());
  Logger::log(Logger::LOG_LEVEL::DEBUG, format("Using {} streams", streams));
  return FilesizeLoadBalancer(input_filenames, output_filenames)
    .partition(streams);
}

auto ColorSearchMain::get_args() const -> const ColorSearchArgumentParser & {
  return *args;
}

auto ColorSearchMain::get_components(
  const shared_ptr<GpuColorIndexContainer> &gpu_container,
  const vector<vector<string>> &split_input_filenames,
  const vector<vector<string>> &split_output_filenames
)
  -> std::tuple<
    vector<shared_ptr<ContinuousIndexFileParser>>,
    vector<shared_ptr<ContinuousColorSearcher>>,
    vector<shared_ptr<ContinuousColorResultsPostProcessor>>,
    vector<shared_ptr<ColorResultsPrinter>>> {
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::START);
  vector<shared_ptr<ContinuousIndexFileParser>> index_file_parsers(streams);
  vector<shared_ptr<ContinuousColorSearcher>> searchers(streams);
  vector<shared_ptr<ContinuousColorResultsPostProcessor>> post_processors(
    streams
  );
  vector<shared_ptr<ColorResultsPrinter>> results_printers(streams);
  for (u64 i = 0; i < streams; ++i) {
    index_file_parsers[i] = make_shared<ContinuousIndexFileParser>(
      i,
      num_components,
      max_indexes_per_batch,
      max_reads_per_batch,
      gpu_warp_size,
      split_input_filenames[i]
    );
    searchers[i] = make_shared<ContinuousColorSearcher>(
      i,
      gpu_container,
      index_file_parsers[i]->get_indexes_batch_producer(),
      max_indexes_per_batch,
      num_components,
      gpu_container->num_colors
    );
    post_processors[i] = make_shared<ContinuousColorResultsPostProcessor>(
      i,
      searchers[i],
      index_file_parsers[i]->get_warps_before_new_read_batch_producer(),
      num_components,
      gpu_container->num_colors
    );
    results_printers[i] = get_results_printer(
      i,
      index_file_parsers[i],
      post_processors[i],
      split_output_filenames[i],
      num_colors
    );
  }
  Logger::log_timed_event("MemoryAllocator", Logger::EVENT_STATE::STOP);
  return {index_file_parsers, searchers, post_processors, results_printers};
}

auto ColorSearchMain::get_results_printer(
  u64 stream_id,
  shared_ptr<ContinuousIndexFileParser> &index_file_parser,
  shared_ptr<ContinuousColorResultsPostProcessor> post_processor,
  const vector<string> &filenames,
  u64 num_colors
) -> shared_ptr<ColorResultsPrinter> {
  if (get_args().get_print_mode() == "ascii") {
    return make_shared<ColorResultsPrinter>(AsciiContinuousColorResultsPrinter(
      stream_id,
      index_file_parser->get_colors_interval_batch_producer(),
      index_file_parser->get_read_statistics_batch_producer(),
      std::move(post_processor),
      filenames,
      num_colors,
      get_args().get_threshold(),
      get_args().get_include_not_found(),
      get_args().get_include_invalid(),
      get_threads(),
      results_printer_max_reads_in_buffer
    ));
  }
  if (get_args().get_print_mode() == "binary") {
    return make_shared<ColorResultsPrinter>(BinaryContinuousColorResultsPrinter(
      stream_id,
      index_file_parser->get_colors_interval_batch_producer(),
      index_file_parser->get_read_statistics_batch_producer(),
      std::move(post_processor),
      filenames,
      num_colors,
      get_args().get_threshold(),
      get_args().get_include_not_found(),
      get_args().get_include_invalid(),
      get_threads(),
      results_printer_max_reads_in_buffer
    ));
  }
  if (get_args().get_print_mode() == "csv") {
    return make_shared<ColorResultsPrinter>(CsvContinuousColorResultsPrinter(
      stream_id,
      index_file_parser->get_colors_interval_batch_producer(),
      index_file_parser->get_read_statistics_batch_producer(),
      std::move(post_processor),
      filenames,
      num_colors,
      get_args().get_threshold(),
      get_args().get_include_not_found(),
      get_args().get_include_invalid(),
      get_threads(),
      results_printer_max_reads_in_buffer
    ));
  }
  throw runtime_error("Invalid value passed by user for argument print_mode");
}

auto ColorSearchMain::run_components(
  vector<shared_ptr<ContinuousIndexFileParser>> &index_file_parsers,
  vector<shared_ptr<ContinuousColorSearcher>> &color_searchers,
  vector<shared_ptr<ContinuousColorResultsPostProcessor>> &post_processors,
  vector<shared_ptr<ColorResultsPrinter>> &results_printers
) -> void {
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::START);
#pragma omp parallel sections num_threads(num_components)
  {
#pragma omp section
    {
#pragma omp parallel for num_threads(streams)
      for (auto &element : index_file_parsers) { element->read_and_generate(); }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(streams)
      for (auto &element : color_searchers) { element->read_and_generate(); }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(streams)
      for (auto &element : post_processors) { element->read_and_generate(); }
    }
#pragma omp section
    {
#pragma omp parallel for num_threads(streams)
      for (auto &element : results_printers) {
        std::visit(
          [](auto &arg) -> void { arg.read_and_generate(); }, *element
        );
      }
    }
  }
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::STOP);
}

}  // namespace sbwt_search
