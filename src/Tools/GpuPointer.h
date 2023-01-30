#ifndef GPU_POINTER_H
#define GPU_POINTER_H

/**
 * @file GpuPointer.h
 * @brief Class to abstract copying to and from the GPU
 */

#include <cstdint>
#include <vector>

namespace gpu_utils {

using std::size_t;
using std::uint8_t;
using std::vector;

template <class T>
class GpuPointer {
private:
  T *ptr;
  size_t bytes = 0;

public:
  explicit GpuPointer(size_t size);
  explicit GpuPointer(const vector<T> &v);
  GpuPointer(const T *cpu_ptr, size_t size);

  GpuPointer(GpuPointer &) = delete;
  GpuPointer(GpuPointer &&) = delete;
  auto operator=(GpuPointer &) = delete;
  auto operator=(GpuPointer &&) = delete;

  auto memset(size_t index, size_t amount, uint8_t value) -> void;

  auto set(const T *source, size_t amount, size_t destination_index = 0)
    -> void;
  auto set(const vector<T> &source, size_t amount, size_t destination_index = 0)
    -> void;
  auto get() const -> T *;

  auto copy_to(T *destination, size_t amount) const -> void;
  auto copy_to(T *destination) const -> void;
  auto copy_to(vector<T> &destination, size_t amount) const -> void;
  auto copy_to(vector<T> &destination) const -> void;

  ~GpuPointer();
};
}  // namespace gpu_utils

#endif
