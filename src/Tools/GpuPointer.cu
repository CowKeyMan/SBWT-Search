#include <iostream>
#include <vector>

#include "Tools/GpuPointer.h"
#include "Tools/GpuUtils.h"
#include "Tools/TypeDefinitions.h"
#include "hip/hip_runtime.h"

using std::vector;

namespace gpu_utils {

template <class T>
GpuPointer<T>::GpuPointer(u64 size): bytes(size * sizeof(T)) {
  GPU_CHECK(hipMalloc((void **)(&ptr), bytes));
}
template <class T>
GpuPointer<T>::GpuPointer(const T *cpu_ptr, u64 size): GpuPointer(size) {
  GPU_CHECK(hipMemcpy(ptr, cpu_ptr, bytes, hipMemcpyHostToDevice));
}
template <class T>
GpuPointer<T>::GpuPointer(const vector<T> &v):
    GpuPointer<T>(v.data(), v.size()) {}

template <class T>
auto GpuPointer<T>::get() const -> T * {
  return ptr;
}

template <class T>
auto GpuPointer<T>::memset(u64 index, u64 amount, uint8_t value) -> void {
  // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
  GPU_CHECK(hipMemset(ptr + index, value, amount * sizeof(T)));
}

template <class T>
auto GpuPointer<T>::set(const T *source, u64 amount, u64 destination_index)
  -> void {
  GPU_CHECK(hipMemcpy(
    // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
    ptr + destination_index,
    source,
    amount * sizeof(T),
    hipMemcpyHostToDevice
  ));
}
template <class T>
auto GpuPointer<T>::set(
  const vector<T> &source, u64 amount, u64 destination_index
) -> void {
  set(source.data(), amount, destination_index);
}

template <class T>
auto GpuPointer<T>::set_async(
  const T *source, u64 amount, GpuStream &gpu_stream, u64 destination_index
) -> void {
  GPU_CHECK(hipMemcpyAsync(
    // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
    ptr + destination_index,
    source,
    amount * sizeof(T),
    hipMemcpyHostToDevice,
    *reinterpret_cast<hipStream_t *>(gpu_stream.get())
  ));
}
template <class T>
auto GpuPointer<T>::set_async(
  const vector<T> &source,
  u64 amount,
  GpuStream &gpu_stream,
  u64 destination_index
) -> void {
  set_async(source.data(), amount, gpu_stream, destination_index);
}

template <class T>
auto GpuPointer<T>::copy_to(T *destination, u64 amount) const -> void {
  GPU_CHECK(
    hipMemcpy(destination, ptr, amount * sizeof(T), hipMemcpyDeviceToHost)
  );
}
template <class T>
auto GpuPointer<T>::copy_to(T *destination) const -> void {
  copy_to(destination, bytes / sizeof(T));
}
template <class T>
auto GpuPointer<T>::copy_to(vector<T> &destination, u64 amount) const -> void {
  copy_to(destination.data(), amount);
}
template <class T>
auto GpuPointer<T>::copy_to(vector<T> &destination) const -> void {
  destination.resize(bytes / sizeof(T));
  copy_to(destination.data());
}

template <class T>
auto GpuPointer<T>::copy_to_async(
  T *destination, u64 amount, GpuStream &gpu_stream
) const -> void {
  GPU_CHECK(hipMemcpyAsync(
    destination,
    ptr,
    amount * sizeof(T),
    hipMemcpyDeviceToHost,
    *reinterpret_cast<hipStream_t *>(gpu_stream.get())
  ));
}
template <class T>
auto GpuPointer<T>::copy_to_async(T *destination, GpuStream &gpu_stream) const
  -> void {
  copy_to_async(destination, bytes / sizeof(T), gpu_stream);
}
template <class T>
auto GpuPointer<T>::copy_to_async(
  vector<T> &destination, u64 amount, GpuStream &gpu_stream
) const -> void {
  copy_to_async(destination.data(), amount, gpu_stream);
}
template <class T>
auto GpuPointer<T>::copy_to_async(vector<T> &destination, GpuStream &gpu_stream)
  const -> void {
  destination.resize(bytes / sizeof(T));
  copy_to_async(destination.data(), gpu_stream);
}

template <class T>
GpuPointer<T>::~GpuPointer() {
  try {
    GPU_CHECK(hipFree(ptr));
  } catch (std::runtime_error &e) { std::cerr << e.what() << std::endl; }
}

// We set these here because we need the class to be instantiated since we are
// using templates
template class GpuPointer<char>;
template class GpuPointer<float>;
template class GpuPointer<double>;
template class GpuPointer<uint64_t>;
template class GpuPointer<int64_t>;
template class GpuPointer<uint32_t>;
template class GpuPointer<int32_t>;
template class GpuPointer<uint16_t>;
template class GpuPointer<uint8_t>;

template class GpuPointer<char *>;
template class GpuPointer<float *>;
template class GpuPointer<double *>;
template class GpuPointer<uint64_t *>;
template class GpuPointer<int64_t *>;
template class GpuPointer<uint32_t *>;
template class GpuPointer<int32_t *>;
template class GpuPointer<uint8_t *>;
template class GpuPointer<uint16_t *>;

}  // namespace gpu_utils
