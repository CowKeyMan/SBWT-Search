#ifndef GPU_EVENT_H
#define GPU_EVENT_H

/**
 * @file GpuEvent.h
 * @brief A class to manage gpu events, useful for not having to create and
 * delete events, and instead we can create them and use them as is.
 */

#include <memory>

#include "Tools/GpuStream.h"

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
  auto record(GpuStream *s = nullptr) -> void;
  // call this function from the start-timer, and give the end-timer as a
  // parameter
  [[nodiscard]] auto get() const -> void *;
  auto time_elapsed_ms(const GpuEvent &e) -> float;
};

}  // namespace gpu_utils

#endif
