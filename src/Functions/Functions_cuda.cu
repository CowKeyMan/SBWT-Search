#include "CudaUtils.cuh"
#include "Functions.h"

namespace functions {

__global__ auto device_add(int *a, int *b, int *out, size_t n) -> void;
__global__ auto device_mul(int *a, int *b, int *out, size_t n) -> void;

auto add(int a, int b) -> int {
  int *device_a = nullptr, *device_b = nullptr;
  cudaMalloc((void **)&device_a, sizeof(int));
  cudaMalloc((void **)&device_b, sizeof(int));
  cudaMemcpy(device_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  device_add<<<1, 1024>>>(device_a, device_b, device_a, 1);
  cudaMemcpy(&a, device_a, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_a);
  cudaFree(device_b);
  return a;
}

__global__ auto device_add(int *a, int *b, int *out, size_t n) -> void {
  auto idx = get_idx();
  if (idx < n) { out[idx] = a[idx] + b[idx]; }
}

auto mul(int a, int b) -> int {
  int *device_a = nullptr, *device_b = nullptr;
  cudaMalloc((void **)&device_a, sizeof(int));
  cudaMalloc((void **)&device_b, sizeof(int));
  cudaMemcpy(device_a, &a, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, &b, sizeof(int), cudaMemcpyHostToDevice);
  device_mul<<<1, 1024>>>(device_a, device_b, device_a, 1);
  cudaMemcpy(&a, device_a, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(device_a);
  cudaFree(device_b);
  return a;
}

__global__ auto device_mul(int *a, int *b, int *out, size_t n) -> void {
  auto idx = get_idx();
  if (idx < n) { out[idx] = a[idx] * b[idx]; }
}

}
