#include "Tools/PinnedVector.h"
#include "hip/hip_runtime.h"

namespace gpu_utils {

template <class T>
PinnedVector<T>::PinnedVector(u64 size): bytes(size * sizeof(T)) {
  // NOLINTNEXTLIE (google-readability-casting)
  hipHostMalloc((void **)(&ptr), size * sizeof(T));
}

template <class T>
auto PinnedVector<T>::data() const -> T * {
  return ptr;
}

template <class T>
auto PinnedVector<T>::operator[](const u64 n) -> T & {
  return ptr[n];  // NOLINT (cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template <class T>
auto PinnedVector<T>::operator[](const u64 n) const -> const T & {
  return ptr[n];  // NOLINT (cppcoreguidelines-pro-bounds-pointer-arithmetic)
}

template <class T>
auto PinnedVector<T>::push_back(const T &elem) -> void {
  // NOLINTNEXTLINE (cppcoreguidelines-pro-bounds-pointer-arithmetic)
  ptr[num_elems] = elem;
  ++num_elems;
}

template <class T>
auto PinnedVector<T>::size() const -> u64 {
  return num_elems;
}

template <class T>
auto PinnedVector<T>::resize(u64 n) -> void {
  num_elems = n;
}

template <class T>
auto PinnedVector<T>::empty() const -> bool {
  return num_elems == 0;
}

// We set these here because we need the class to be instantiated since we are
// using templates
template class PinnedVector<char>;
template class PinnedVector<float>;
template class PinnedVector<double>;
template class PinnedVector<uint64_t>;
template class PinnedVector<int64_t>;
template class PinnedVector<uint32_t>;
template class PinnedVector<int32_t>;
template class PinnedVector<uint16_t>;
template class PinnedVector<uint8_t>;

template class PinnedVector<char *>;
template class PinnedVector<float *>;
template class PinnedVector<double *>;
template class PinnedVector<uint64_t *>;
template class PinnedVector<int64_t *>;
template class PinnedVector<uint32_t *>;
template class PinnedVector<int32_t *>;
template class PinnedVector<uint8_t *>;
template class PinnedVector<uint16_t *>;
}  // namespace gpu_utils
