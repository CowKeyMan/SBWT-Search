#include <algorithm>
#include <iostream>
#include <limits>
#include <span>
#include <vector>
using std::cerr;
using std::cout;
using std::endl;
using std::vector;
template <class T>
auto print_vec(
  const vector<T> &v, uint64_t limit = std::numeric_limits<uint64_t>::max()
) {
  cout << "---------------------" << endl;
  for (int i = 0; i < std::min(limit, v.size()); ++i) { cout << v[i] << " "; }
  cout << endl << "---------------------" << endl;
}

#include <omp.h>
#include <stdexcept>
#include <string>

#include "ArgumentParser/ColorSearchArgumentParser.h"
#include "ColorIndexBuilder/ColorIndexBuilder.h"
#include "Global/GlobalDefinitions.h"
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
  auto [index_file_parser, searcher, post_processor, results_printer]
    = get_components(
      gpu_container,
      get_input_filenames(),
      get_output_filenames(),
      args.get_print_mode(),
      args.get_threshold()
    );
  Logger::log(Logger::LOG_LEVEL::INFO, "Running queries");
  run_components(index_file_parser, searcher, post_processor, results_printer);
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
  const vector<string> &input_filenames,
  const vector<string> &output_filenames,
  const string &print_mode,
  double threshold
)
  -> std::tuple<
    shared_ptr<ContinuousIndexFileParser>,
    shared_ptr<ContinuousColorSearcher>,
    shared_ptr<ContinuousColorResultsPostProcessor>,
    shared_ptr<ContinuousColorSearchResultsPrinter>> {
  auto index_file_parser = make_shared<ContinuousIndexFileParser>(
    max_batches, max_indexes_per_batch, gpu_warp_size, input_filenames
  );
  auto searcher = make_shared<ContinuousColorSearcher>(
    gpu_container,
    index_file_parser->get_indexes_batch_producer(),
    max_indexes_per_batch,
    max_batches,
    gpu_container->num_colors
  );
  auto post_processor = make_shared<ContinuousColorResultsPostProcessor>(
    searcher,
    index_file_parser->get_warps_before_new_read_batch_producer(),
    max_batches,
    gpu_container->num_colors
  );
  auto results_printer = make_shared<ContinuousColorSearchResultsPrinter>(
    index_file_parser->get_colors_interval_batch_producer(),
    index_file_parser->get_read_statistics_batch_producer(),
    post_processor,
    output_filenames,
    gpu_container->num_colors,
    threshold
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
  const u64 sections = 4;
#pragma omp parallel sections num_threads(sections)
  {
#pragma omp section
    {
      cout << __LINE__ << endl;
      index_file_parser->read_and_generate();
      cout << __LINE__ << endl;
    }
#pragma omp section
    {
      cout << __LINE__ << endl;
      color_searcher->read_and_generate();
      cout << __LINE__ << endl;
    }
#pragma omp section
    {
      cout << __LINE__ << endl;
      post_processor->read_and_generate();
      cout << __LINE__ << endl;
    }
#pragma omp section
    {
      cout << __LINE__ << endl;
      results_printer->read_and_generate();
      cout << __LINE__ << endl;
    }
  }
  Logger::log_timed_event("Querier", Logger::EVENT_STATE::STOP);
}

}  // namespace sbwt_search
