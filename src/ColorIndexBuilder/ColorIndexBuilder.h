#ifndef COLOR_INDEX_BUILDER_H
#define COLOR_INDEX_BUILDER_H

/**
 * @file ColorIndexBuilder.h
 * @brief Here, the colors file is read from disk and the CpuColorIndexContainer
 * is constructed from it
 */

#include <string>

#include "ColorIndexContainer/CpuColorIndexContainer.h"
#include "Tools/IOUtils.h"

namespace sbwt_search {

using io_utils::ThrowingIfstream;
using std::string;

class ColorIndexBuilder {
private:
  ThrowingIfstream in_stream;

public:
  explicit ColorIndexBuilder(const string &filename);

  auto get_cpu_color_index_container() -> CpuColorIndexContainer;
};

}  // namespace sbwt_search

#endif
