#include <vector>

#include "Tools/GpuPointer.h"
#include "Tools/GpuUtils.h"
#include "Tools/TypeDefinitions.h"
#include "hip/hip_runtime.h"

using std::vector;

namespace gpu_utils {

template <class T>
GpuPointer<T>::GpuPointer(u64 size): bytes(size * sizeof(T)) {
  GPU_CHECK(hipMalloc(&ptr, bytes));
}
template <class T>
GpuPointer<T>::GpuPointer(const T *cpu_ptr, u64 size): GpuPointer(size) {
  GPU_CHECK(hipMemcpy(ptr, cpu_ptr, bytes, hipMemcpyHostToDevice));
}
template <class T>
GpuPointer<T>::GpuPointer(const vector<T> &v):
    GpuPointer<T>(v.data(), v.size()) {}

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
auto GpuPointer<T>::get() const -> T * {
  return ptr;
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
GpuPointer<T>::~GpuPointer() {
  hipFree(ptr);
}

// We set these here because we need the class to be instantiated since we are
// using templates
template class GpuPointer<float>;
template class GpuPointer<double>;
template class GpuPointer<uint64_t>;
template class GpuPointer<int64_t>;
template class GpuPointer<uint32_t>;
template class GpuPointer<int32_t>;

template class GpuPointer<float *>;
template class GpuPointer<double *>;
template class GpuPointer<uint64_t *>;
template class GpuPointer<int64_t *>;
template class GpuPointer<uint32_t *>;
template class GpuPointer<int32_t *>;

}  // namespace gpu_utils
