#ifndef GPU_UTILS_H
#define GPU_UTILS_H

/**
 * @file GpuUtils.h
 * @brief Contains CUDA commonly used functions and tools
 */

#include <sstream>

#include <cuda_runtime.h>

#include "Tools/TypeDefinitions.h"

namespace gpu_utils {

using std::runtime_error;
using std::stringstream;

template <class Error_t>
auto getErrorString(Error_t code) -> const char *;

template <class Error_t>
inline auto gpuAssert(Error_t code, const char *file, int line) -> void {
  if (code != cudaSuccess) {
    stringstream ss;
    ss << "GPUassert: " << getErrorString(code) << " at " << file << ":" << line
       << '\n';
    throw runtime_error(ss.str());
  }
}

// This is kept as a macro instead of converting it to a modern c++ constexpr
// because otherwise __FILE__ and __LINE__ will not work as intended, ie to
// report where the error is
// NOLINTNEXTLINE(cppcoreguidelines-macro-usage)
#define GPU_CHECK(code_block) \
  { gpu_utils::gpuAssert((code_block), __FILE__, __LINE__); }

auto get_free_gpu_memory() -> u64;

}  // namespace gpu_utils

#endif
