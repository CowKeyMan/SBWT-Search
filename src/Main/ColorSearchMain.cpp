#include <omp.h>
#include <stdexcept>
#include <string>

#include "ArgumentParser/ColorSearchArgumentParser.h"
#include "ColorIndexBuilder/ColorIndexBuilder.h"
#include "FilenamesParser/FilenamesParser.h"
#include "Global/GlobalDefinitions.h"
#include "Main/ColorSearchMain.h"
#include "Tools/GpuUtils.h"
#include "Tools/Logger.h"
#include "Tools/MathUtils.hpp"
#include "Tools/MemoryUtils.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using gpu_utils::get_free_gpu_memory;
using log_utils::Logger;
using math_utils::bits_to_gB;
using math_utils::round_down;
using memory_utils::get_total_system_memory;
using std::min;
using std::runtime_error;
using std::to_string;

const u64 num_components = 4;

auto ColorSearchMain::main(int argc, char **argv) -> int {
  const string program_name = "colors";
  const string program_description = "sbwt_search";
  Logger::log_timed_event("main", Logger::EVENT_STATE::START);
  args = make_unique<ColorSearchArgumentParser>(
    program_name, program_description, argc, argv
  );
  Logger::log(Logger::LOG_LEVEL::INFO, "Loading components into memory");
  auto gpu_container = get_gpu_container();
  num_colors = gpu_container->num_colors;
  auto [input_filenames, output_filenames] = get_input_output_filenames();
  load_batch_info();
  omp_set_nested(1);
  load_threads();
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    format("Running OpenMP with {} threads", get_threads())
  );
  auto [index_file_parser, searcher, post_processor, results_printer]
    = get_components(gpu_container, input_filenames[0], output_filenames[0]);
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
  auto gpu_chars = get_max_chars_per_batch_gpu();
  auto cpu_chars = get_max_chars_per_batch_cpu();
  return min(gpu_chars, cpu_chars);
}

auto ColorSearchMain::get_max_chars_per_batch_gpu() -> u64 {
  u64 free = static_cast<u64>(
    static_cast<double>(get_free_gpu_memory() * sizeof(u64))
    * get_args().get_gpu_memory_percentage()
  );
  // 64 for each index found
  // The results take: num_colors * 64 / warp_size
  const u64 bits_required_per_character = static_cast<u64>(
    64.0
    + static_cast<double>(num_colors) * 64.0
      / static_cast<double>(gpu_warp_size)
  );
  auto max_chars_per_batch
    = round_down<u64>(free / bits_required_per_character, threads_per_block);
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
  // 64 bits per index
  // 64 * num_colors / warp_size to store the results
  // 20 * 8 bits for each printed character
  // 1 * 8 bits for each space after the printed characters
  // 64 bits per read for each newline
  // 64 bits per read for each found result count
  // 64 bits per read for each not found result count
  // 64 bits per read for each invalid result count
  const u64 bits_required_per_character = static_cast<u64>(
    64.0
    + 64.0 * static_cast<double>(num_colors)
      / static_cast<double>(gpu_warp_size)
    + (20 + 1) * 8
    + (1.0 / static_cast<double>(get_args().get_indexes_per_read()))
      * (64.0 * 4)
  );
  auto max_chars_per_batch = round_down<u64>(
    free_bits / bits_required_per_character / num_components, threads_per_block
  );
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
  auto all_input_filenames = filenames_parser.get_input_filenames();
  auto all_output_filenames = filenames_parser.get_output_filenames();
  if (all_input_filenames.size() != all_output_filenames.size()) {
    throw runtime_error("Input and output file sizes differ");
  }
  return {{all_input_filenames}, {all_output_filenames}};
}

auto ColorSearchMain::get_args() const -> const ColorSearchArgumentParser & {
  return *args;
}

auto ColorSearchMain::get_components(
  const shared_ptr<GpuColorIndexContainer> &gpu_container,
  const vector<string> &input_filenames,
  const vector<string> &output_filenames
)
  -> std::tuple<
    shared_ptr<ContinuousIndexFileParser>,
    shared_ptr<ContinuousColorSearcher>,
    shared_ptr<ContinuousColorResultsPostProcessor>,
    shared_ptr<ContinuousColorSearchResultsPrinter>> {
  auto index_file_parser = make_shared<ContinuousIndexFileParser>(
    num_components,
    max_indexes_per_batch,
    max_reads_per_batch,
    gpu_warp_size,
    input_filenames
  );
  auto searcher = make_shared<ContinuousColorSearcher>(
    gpu_container,
    index_file_parser->get_indexes_batch_producer(),
    max_indexes_per_batch,
    num_components,
    gpu_container->num_colors
  );
  auto post_processor = make_shared<ContinuousColorResultsPostProcessor>(
    searcher,
    index_file_parser->get_warps_before_new_read_batch_producer(),
    num_components,
    gpu_container->num_colors
  );
  auto results_printer = make_shared<ContinuousColorSearchResultsPrinter>(
    index_file_parser->get_colors_interval_batch_producer(),
    index_file_parser->get_read_statistics_batch_producer(),
    post_processor,
    output_filenames,
    gpu_container->num_colors,
    get_args().get_threshold(),
    get_args().get_include_not_found(),
    get_args().get_include_invalid()
  );
  return {index_file_parser, searcher, post_processor, results_printer};
}

auto ColorSearchMain::run_components(
  shared_ptr<ContinuousIndexFileParser> &index_file_parser,
  shared_ptr<ContinuousColorSearcher> &color_searcher,
  shared_ptr<ContinuousColorResultsPostProcessor> &post_processor,
  shared_ptr<ContinuousColorSearchResultsPrinter> &results_printer
) -> void {
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::START);
#pragma omp parallel sections num_threads(num_components)
  {
#pragma omp section
    { index_file_parser->read_and_generate(); }
#pragma omp section
    { color_searcher->read_and_generate(); }
#pragma omp section
    { post_processor->read_and_generate(); }
#pragma omp section
    { results_printer->read_and_generate(); }
  }
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::STOP);
}

}  // namespace sbwt_search
