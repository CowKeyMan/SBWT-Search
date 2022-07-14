#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/**
 * @file CudaUtils.cuh
 * @brief Contains CUDA commonly used functions and tools
 * */

#include <cstddef>
#include <iostream>
#include <vector>

using std::cerr;
using std::endl;
using std::vector;

auto get_idx() -> int { return blockDim.x * blockIdx.x + threadIdx.x; }

#define CUDA_CHECK(code_block) \
  { gpuAssert((code_block), __FILE__, __LINE__); }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    cerr << "GPUassert: " << cudaGetErrorString(code) << " " << line << " "
         << line << '\n';
    if (abort) exit(code);
  }
}

template <class T>
class CudaPointer {
    T *ptr;

  public:
    CudaPointer(size_t size) {
      mysize = size * sizeof(T);
      CUDA_CHECK(cudaMalloc((void **)&ptr, mysize));
    }
    CudaPointer(const T *cpu_ptr, size_t size): CudaPointer(size) {
      CUDA_CHECK(cudaMemcpy(ptr, cpu_ptr, mysize, cudaMemcpyHostToDevice));
    }
    CudaPointer(const vector<T> v): CudaPointer(&v[0], v.size()) {}
    auto set(const T elem) -> void {
      CUDA_CHECK(cudaMemset(ptr, elem, mysize));
    }
    auto get() const -> T *const { return ptr; }
    auto copy_to(T &destination) -> void {
      CUDA_CHECK(cudaMemcpy(&destination, ptr, mysize, cudaMemcpyDeviceToHost));
    }
    auto copy_to(vector<T> &destination) -> void {
      CUDA_CHECK(
        cudaMemcpy(&destination[0], ptr, mysize, cudaMemcpyDeviceToHost)
      );
    }
    ~CudaPointer() { CUDA_CHECK(cudaFree(ptr)); }

  private:
    size_t mysize;
};

#endif
