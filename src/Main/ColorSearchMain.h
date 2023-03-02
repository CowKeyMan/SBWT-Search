#ifndef COLOR_SEARCH_MAIN_H
#define COLOR_SEARCH_MAIN_H

/**
 * @file ColorSearchMain.h
 * @brief The main function for searching for the colors. The 'color' mode of
 * the main executable
 */

#include <memory>
#include <string>

#include "ArgumentParser/ColorSearchArgumentParser.h"
#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "ColorResultsPostProcessor/ContinuousColorResultsPostProcessor.h"
#include "ColorSearchResultsPrinter/ContinuousColorSearchResultsPrinter.hpp"
#include "ColorSearcher/ContinuousColorSearcher.h"
#include "IndexFileParser/ContinuousIndexFileParser.h"
#include "Main/Main.h"

namespace sbwt_search {

using std::shared_ptr;
using std::string;

class ColorSearchMain: public Main {
private:
  u64 max_indexes_per_batch = 0;
  u64 num_colors = 0;
  unique_ptr<ColorSearchArgumentParser> args;

public:
  auto main(int argc, char **argv) -> int override;

private:
  [[nodiscard]] auto get_args() const -> const ColorSearchArgumentParser &;
  auto get_gpu_container() -> shared_ptr<GpuColorIndexContainer>;
  auto load_batch_info() -> void;
  auto get_max_chars_per_batch_cpu() -> u64;
  auto get_max_chars_per_batch_gpu() -> u64;
  auto get_max_chars_per_batch() -> u64;
  auto get_input_output_filenames()
    -> std::tuple<vector<vector<string>>, vector<vector<string>>>;
  auto get_components(
    const shared_ptr<GpuColorIndexContainer> &gpu_container,
    const vector<string> &input_filenames,
    const vector<string> &output_filenames
  )
    -> std::tuple<
      shared_ptr<ContinuousIndexFileParser>,
      shared_ptr<ContinuousColorSearcher>,
      shared_ptr<ContinuousColorResultsPostProcessor>,
      shared_ptr<ContinuousColorSearchResultsPrinter>>;
  auto run_components(
    shared_ptr<ContinuousIndexFileParser> &index_file_parser,
    shared_ptr<ContinuousColorSearcher> &color_searcher,
    shared_ptr<ContinuousColorResultsPostProcessor> &post_processor,
    shared_ptr<ContinuousColorSearchResultsPrinter> &results_processor
  ) -> void;
};

}  // namespace sbwt_search

#endif
