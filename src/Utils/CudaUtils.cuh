#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

/**
 * @file CudaUtils.cuh
 * @brief Contains CUDA commonly used functions and tools
 * */

#include "cuda_runtime.h"

__device__ auto get_idx() -> int {
  return blockDim.x * blockIdx.x + threadIdx.x;
}

#define CUDA_CHECK(code_block) \
  { gpuAssert((code_block), __FILE__, __LINE__); }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(
      stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line
    );
    if (abort) exit(code);
  }
}

template <class T>
class CUDA_ptr {
    T *ptr;

  public:
    CUDA_ptr(size_t size) {
      mysize = size * sizeof(T);
      CUDA_CHECK(cudaMalloc((void **)&ptr, mysize));
    }
    CUDA_ptr(const vector<T> v): CUDA_ptr(v.size()) {
      CUDA_CHECK(cudaMemcpy(ptr, &v[0], mysize, cudaMemcpyHostToDevice));
    }
    auto set(const T elem) -> void {
      CUDA_CHECK(cudaMemset(ptr, elem, mysize));
    }
    auto get() -> T * { return ptr; }
    auto copy_to(T &destination) -> void {
      CUDA_CHECK(cudaMemcpy(&destination, ptr, mysize, cudaMemcpyDeviceToHost));
    }
    auto copy_to(vector<T> &destination) -> void {
      CUDA_CHECK(
        cudaMemcpy(&destination[0], ptr, mysize, cudaMemcpyDeviceToHost)
      );
    }
    ~CUDA_ptr() { CUDA_CHECK(cudaFree(ptr)); }

  private:
    size_t mysize;
};

#endif
