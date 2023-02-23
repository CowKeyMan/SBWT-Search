#include <omp.h>
#include <stdexcept>
#include <string>

#include "ArgumentParser/ColorSearchArgumentParser.h"
#include "ColorIndexBuilder/ColorIndexBuilder.h"
#include "Main/ColorSearchMain.h"
#include "Tools/Logger.h"
#include "fmt/core.h"

namespace sbwt_search {

using fmt::format;
using log_utils::Logger;
using std::runtime_error;
using std::to_string;

auto ColorSearchMain::main(int argc, char **argv) -> int {
  const string program_name = "colors";
  const string program_description = "sbwt_search";
  Logger::log_timed_event("main", Logger::EVENT_STATE::START);
  auto args
    = ColorSearchArgumentParser(program_name, program_description, argc, argv);
  Logger::log(Logger::LOG_LEVEL::INFO, "Loading components into memory");
  auto gpu_container = get_gpu_container(args.get_colors_file());
  load_input_output_filenames(args.get_query_file(), args.get_output_file());
  load_batch_info(
    args.get_batches(), args.get_unavailable_ram(), args.get_max_cpu_memory()
  );
  omp_set_nested(1);
  load_threads();
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    format("Running OpenMP with {} threads", get_threads())
  );
  auto [index_file_parser, searcher]
    = get_components(gpu_container, args.get_print_mode());
  Logger::log(Logger::LOG_LEVEL::INFO, "Running queries");
  run_components(index_file_parser, searcher);
  Logger::log(Logger::LOG_LEVEL::INFO, "Finished");
  Logger::log_timed_event("main", Logger::EVENT_STATE::STOP);
  return 0;
}

inline auto ColorSearchMain::get_gpu_container(const string &colors_filename)
  -> shared_ptr<GpuColorIndexContainer> {
  Logger::log_timed_event("ColorsLoader", Logger::EVENT_STATE::START);
  auto color_index_builder = ColorIndexBuilder(colors_filename);
  auto cpu_container = color_index_builder.get_cpu_color_index_container();
  auto gpu_container = cpu_container.to_gpu();
  Logger::log_timed_event("ColorsLoader", Logger::EVENT_STATE::STOP);
  return gpu_container;
}

auto ColorSearchMain::load_batch_info(
  u64 max_batches_, u64 unavailable_ram, u64 max_cpu_memory
) -> void {
  max_batches = max_batches_;
  max_indexes_per_batch = 1024;  // TODO: turn into a function, calculate c/gpu
  if (max_indexes_per_batch == 0) { throw runtime_error("Not enough memory"); }
  Logger::log(
    Logger::LOG_LEVEL::INFO,
    "Using " + to_string(max_indexes_per_batch) + " characters per batch"
  );
}

auto ColorSearchMain::get_components(
  const shared_ptr<GpuColorIndexContainer> &gpu_container,
  const string &print_mode
)
  -> std::tuple<
    shared_ptr<ContinuousIndexFileParser>,
    shared_ptr<ContinuousColorSearcher>> {
  const u64 read_padding = 32;
  auto index_file_parser = make_shared<ContinuousIndexFileParser>(
    max_indexes_per_batch, max_batches, get_input_filenames(), read_padding
  );
  auto searcher = make_shared<ContinuousColorSearcher>(
    gpu_container,
    index_file_parser->get_indexes_batch_producer(),
    max_indexes_per_batch,
    max_batches
  );
  return {index_file_parser, searcher};
}

auto ColorSearchMain::run_components(
  shared_ptr<ContinuousIndexFileParser> &index_file_parser,
  shared_ptr<ContinuousColorSearcher> &color_searcher
) -> void {
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::START);
  const u64 sections = 2;
#pragma omp parallel sections num_threads(sections)
  {
#pragma omp section
    { index_file_parser->read_and_generate(); }
#pragma omp section
    { color_searcher->read_and_generate(); }
  }
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::STOP);
}

}  // namespace sbwt_search
