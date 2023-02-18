#ifndef GPU_UTILS_H
#define GPU_UTILS_H

/**
 * @file GpuUtils.h
 * @brief Contains GPU commonly used functions and tools
 */

#include <sstream>
#include <stdexcept>

#include "Tools/TypeDefinitions.h"

namespace gpu_utils {

// This is kept as a macro instead of converting it to a modern c++ constexpr
// because otherwise __FILE__ and __LINE__ will not work as intended, ie to
// report where the error is
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GPU_CHECK(code_block)                                              \
  {                                                                        \
    auto code = code_block;                                                \
    if (code != hipSuccess) {                                              \
      std::stringstream ss;                                                \
      ss << "GPUassert: " << hipGetErrorString(code) << " at " << __FILE__ \
         << ":" << __LINE__ << '\n';                                       \
      throw std::runtime_error(ss.str());                                  \
    }                                                                      \
  }

auto get_free_gpu_memory() -> u64;

}  // namespace gpu_utils

#endif
