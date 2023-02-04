#include <vector>

#include "Tools/GpuPointer.h"
#include "Tools/GpuUtils.h"

using std::vector;

namespace gpu_utils {

template <class T>
GpuPointer<T>::GpuPointer(size_t size): bytes(size * sizeof(T)) {
  GPU_CHECK(cudaMalloc(&ptr, bytes));
}
template <class T>
GpuPointer<T>::GpuPointer(const T *cpu_ptr, size_t size): GpuPointer(size) {
  GPU_CHECK(cudaMemcpy(ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice));
}
template <class T>
GpuPointer<T>::GpuPointer(const vector<T> &v):
    GpuPointer<T>(v.data(), v.size()) {}

template <class T>
auto GpuPointer<T>::memset(size_t index, size_t amount, uint8_t value) -> void {
  // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
  GPU_CHECK(cudaMemset(ptr + index, value, amount * sizeof(T)));
}

template <class T>
auto GpuPointer<T>::set(
  const T *source, size_t amount, size_t destination_index
) -> void {
  GPU_CHECK(cudaMemcpy(
    // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
    ptr + destination_index,
    source,
    amount * sizeof(T),
    cudaMemcpyHostToDevice
  ));
}
template <class T>
auto GpuPointer<T>::set(
  const vector<T> &source, size_t amount, size_t destination_index
) -> void {
  set(source.data(), amount, destination_index);
}

template <class T>
auto GpuPointer<T>::get() const -> T * {
  return ptr;
}

template <class T>
auto GpuPointer<T>::copy_to(T *destination, size_t amount) const -> void {
  GPU_CHECK(
    cudaMemcpy(destination, ptr, amount * sizeof(T), cudaMemcpyDeviceToHost)
  );
}
template <class T>
auto GpuPointer<T>::copy_to(T *destination) const -> void {
  copy_to(destination, bytes / sizeof(T));
}
template <class T>
auto GpuPointer<T>::copy_to(vector<T> &destination, size_t amount) const
  -> void {
  copy_to(destination.data(), amount);
}
template <class T>
auto GpuPointer<T>::copy_to(vector<T> &destination) const -> void {
  copy_to(destination.data());
}

template <class T>
GpuPointer<T>::~GpuPointer() {
  cudaFree(ptr);
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
