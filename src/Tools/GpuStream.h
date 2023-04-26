#ifndef GPU_STREAM_H
#define GPU_STREAM_H

/**
 * @file GpuStream.h
 * @brief A class to manage gpu streams, useful for not having to create and
 * delete streams, and instead we can create them and use them as is.
 */

#include <memory>

namespace gpu_utils {

using std::shared_ptr;

class GpuStream {
  void *element;

public:
  GpuStream();
  GpuStream(GpuStream &) = delete;
  GpuStream(GpuStream &&) = delete;
  auto operator=(GpuStream &) = delete;
  auto operator=(GpuStream &&) = delete;
  ~GpuStream();
  [[nodiscard]] auto data() const -> void *;
};

}  // namespace gpu_utils

#endif
