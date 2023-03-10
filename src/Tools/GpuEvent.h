#ifndef GPU_EVENT_H
#define GPU_EVENT_H

/**
 * @file GpuEvent.h
 * @brief A class to manage gpu events, useful for not having to create and
 * delete events, and instead we can create them and use them as is.
 */

#include <memory>

namespace gpu_utils {

using std::shared_ptr;

class GpuEvent {
  void *element;

public:
  GpuEvent();
  GpuEvent(GpuEvent &) = delete;
  GpuEvent(GpuEvent &&) = delete;
  auto operator=(GpuEvent &) = delete;
  auto operator=(GpuEvent &&) = delete;
  ~GpuEvent();
  [[nodiscard]] auto get() const -> void *;
};

}  // namespace gpu_utils

#endif
