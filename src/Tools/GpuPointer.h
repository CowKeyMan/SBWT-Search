#ifndef GPU_POINTER_H
#define GPU_POINTER_H

/**
 * @file GpuPointer.h
 * @brief Class to abstract copying to and from the GPU
 */

#include <cstdint>
#include <vector>

#include "Tools/GpuStream.h"
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

  GpuPointer(u64 size, GpuStream &gpu_stream);
  GpuPointer(const vector<T> &v, GpuStream &gpu_stream);
  GpuPointer(const T *cpu_ptr, u64 size, GpuStream &gpu_stream);

  GpuPointer(GpuPointer &) = delete;
  GpuPointer(GpuPointer &&) = delete;
  auto operator=(GpuPointer &) = delete;
  auto operator=(GpuPointer &&) = delete;

  auto memset(u64 index, u64 amount, uint8_t value) -> void;
  auto memset(u64 index, uint8_t value) -> void;
  auto memset_async(u64 index, u64 amount, uint8_t value, GpuStream &gpu_stream)
    -> void;
  auto memset_async(u64 index, uint8_t value, GpuStream &gpu_stream) -> void;

  auto get() const -> T *;

  auto set(const T *source, u64 amount, u64 destination_index = 0) -> void;
  auto set(const vector<T> &source, u64 amount, u64 destination_index = 0)
    -> void;
  auto set_async(
    const T *source,
    u64 amount,
    GpuStream &gpu_stream,
    u64 destination_index = 0
  ) -> void;
  auto set_async(
    const vector<T> &source,
    u64 amount,
    GpuStream &gpu_stream,
    u64 destination_index = 0
  ) -> void;

  auto copy_to(T *destination, u64 amount) const -> void;
  auto copy_to(T *destination) const -> void;
  auto copy_to(vector<T> &destination, u64 amount) const -> void;
  auto copy_to(vector<T> &destination) const -> void;

  auto copy_to_async(T *destination, u64 amount, GpuStream &gpu_stream) const
    -> void;
  auto copy_to_async(T *destination, GpuStream &gpu_stream) const -> void;
  auto
  copy_to_async(vector<T> &destination, u64 amount, GpuStream &gpu_stream) const
    -> void;
  auto copy_to_async(vector<T> &destination, GpuStream &gpu_stream) const
    -> void;

  ~GpuPointer();
};
}  // namespace gpu_utils

#endif
