#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/**
 * @file CudaUtils.cuh
 * @brief Contains CUDA commonly used functions and tools
 * */

#include <cstddef>
#include <iostream>
#include <vector>

namespace gpu_utils {

using std::cerr;
using std::endl;
using std::vector;

inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    cerr << "GPUassert: " << cudaGetErrorString(code) << " at " << file << ":"
         << line << '\n';
    if (abort) exit(code);
  }
}
#define GPU_CHECK(code_block) \
  { gpu_utils::gpuAssert((code_block), __FILE__, __LINE__); }

template <class T>
class GpuPointer {
  private:
    T *ptr;
    size_t bytes;

  public:
    GpuPointer(size_t size): bytes(size * sizeof(T)) {
      GPU_CHECK(cudaMalloc((void **)&ptr, bytes));
    }
    GpuPointer(const T *cpu_ptr, size_t size): GpuPointer(size) {
      GPU_CHECK(cudaMemcpy(ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice));
    }
    GpuPointer(const vector<T> &v): GpuPointer(&v[0], v.size()) {}

    auto memset(const size_t index, const size_t amount, const uint8_t value)
      -> void {
      GPU_CHECK(cudaMemset(ptr + index, value, amount * sizeof(T)));
    }

    auto set(
      const T *source, const size_t amount, const size_t destination_index = 0
    ) -> void {
      GPU_CHECK(cudaMemcpy(
        ptr + destination_index,
        source,
        amount * sizeof(T),
        cudaMemcpyHostToDevice
      ));
    }
    auto set(
      const vector<T> &source,
      const size_t amount,
      const size_t destination_index = 0
    ) -> void {
      set(&source[0], amount, destination_index);
    }

    auto get() const -> T *const { return ptr; }

    auto copy_to(T *destination, const size_t amount) const -> void {
      GPU_CHECK(
        cudaMemcpy(destination, ptr, amount * sizeof(T), cudaMemcpyDeviceToHost)
      );
    }
    auto copy_to(T *destination) const -> void {
      copy_to(destination, bytes / sizeof(T));
    }
    auto copy_to(vector<T> &destination, const size_t amount) const -> void {
      copy_to(&destination[0], amount);
    }
    auto copy_to(vector<T> &destination) const -> void {
      copy_to(&destination[0]);
    }

    ~GpuPointer() { GPU_CHECK(cudaFree(ptr)); }
};

auto get_free_gpu_memory() -> size_t;

}  // namespace gpu_utils

#endif
