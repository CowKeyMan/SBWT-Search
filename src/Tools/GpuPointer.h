#ifndef GPU_POINTER_H
#define GPU_POINTER_H

/**
 * @file GpuPointer.h
 * @brief Class to abstract copying to and from the GPU
 */

#include <cstdint>
#include <vector>

#include "Tools/TypeDefinitions.h"

namespace gpu_utils {

using std::uint8_t;
using std::vector;

template <class T>
class GpuPointer {
private:
  T *ptr;
  u64 bytes = 0;

public:
  explicit GpuPointer(u64 size);
  explicit GpuPointer(const vector<T> &v);
  GpuPointer(const T *cpu_ptr, u64 size);

  GpuPointer(GpuPointer &) = delete;
  GpuPointer(GpuPointer &&) = delete;
  auto operator=(GpuPointer &) = delete;
  auto operator=(GpuPointer &&) = delete;

  auto memset(u64 index, u64 amount, uint8_t value) -> void;

  auto set(const T *source, u64 amount, u64 destination_index = 0) -> void;
  auto set(const vector<T> &source, u64 amount, u64 destination_index = 0)
    -> void;
  auto get() const -> T *;

  auto copy_to(T *destination, u64 amount) const -> void;
  auto copy_to(T *destination) const -> void;
  auto copy_to(vector<T> &destination, u64 amount) const -> void;
  auto copy_to(vector<T> &destination) const -> void;

  ~GpuPointer();
};
}  // namespace gpu_utils

#endif
