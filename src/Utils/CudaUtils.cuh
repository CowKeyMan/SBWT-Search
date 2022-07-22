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

#define CUDA_CHECK(code_block) \
  { gpuAssert((code_block), __FILE__, __LINE__); }
inline void
gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    cerr << "GPUassert: " << cudaGetErrorString(code) << " at " << file << ":"
         << line << '\n';
    if (abort) exit(code);
  }
}

#include <iostream>
using namespace std;

template <class T>
class CudaPointer {
  private:
    T *ptr;
    size_t bytes;

  public:
    CudaPointer(size_t size): bytes(size * sizeof(T)) {
      CUDA_CHECK(cudaMalloc((void **)&ptr, bytes));
    }
    CudaPointer(const T *cpu_ptr, size_t size): CudaPointer(size) {
      CUDA_CHECK(cudaMemcpy(ptr, cpu_ptr, bytes, cudaMemcpyHostToDevice));
    }
    CudaPointer(const vector<T> &v): CudaPointer(&v[0], v.size()) {}
    auto set(const T elem) -> void { CUDA_CHECK(cudaMemset(ptr, elem, bytes)); }
    auto get() const -> T *const { return ptr; }
    auto copy_to(T &destination) const -> void {
      CUDA_CHECK(cudaMemcpy(&destination, ptr, bytes, cudaMemcpyDeviceToHost));
    }
    auto copy_to(vector<T> &destination) const -> void {
      CUDA_CHECK(cudaMemcpy(&destination[0], ptr, bytes, cudaMemcpyDeviceToHost)
      );
    }
    ~CudaPointer() { CUDA_CHECK(cudaFree(ptr)); }
};

#endif
