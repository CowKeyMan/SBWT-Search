#ifndef COLOR_SEARCH_MAIN_H
#define COLOR_SEARCH_MAIN_H

/**
 * @file ColorSearchMain.h
 * @brief The main function for searching for the colors. The 'color' mode of
 * the main executable
 */

#include <memory>
#include <string>

#include "ColorIndexContainer/GpuColorIndexContainer.h"
#include "ColorSearcher/ContinuousColorSearcher.h"
#include "IndexFileParser/ContinuousIndexFileParser.h"
#include "Main/Main.h"

namespace sbwt_search {

using std::shared_ptr;
using std::string;

class ColorSearchMain: public Main {
private:
  u64 max_indexes_per_batch = 0;
  u64 max_batches = 0;

public:
  auto main(int argc, char **argv) -> int override;

private:
  auto get_gpu_container(const string &colors_filename)
    -> shared_ptr<GpuColorIndexContainer>;
  auto
  load_batch_info(u64 max_batches_, u64 unavailable_ram, u64 max_cpu_memory)
    -> void;
  auto get_components(
    const shared_ptr<GpuColorIndexContainer> &gpu_container,
    const string &print_mode
  )
    -> std::tuple<
      shared_ptr<ContinuousIndexFileParser>,
      shared_ptr<ContinuousColorSearcher>>;
};

}  // namespace sbwt_search

#endif
