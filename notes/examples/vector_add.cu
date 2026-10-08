#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

void check(cudaError_t result, const char* step) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1000;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 2.0f;
  }

  float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
  check(cudaMalloc(&d_a, bytes), "cudaMalloc a");
  check(cudaMalloc(&d_b, bytes), "cudaMalloc b");
  check(cudaMalloc(&d_c, bytes), "cudaMalloc c");
  check(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "copy a");
  check(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "copy b");

  const int threadsPerBlock = 256;
  const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
  vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);
  check(cudaGetLastError(), "kernel launch");
  check(cudaDeviceSynchronize(), "kernel execution");

  check(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "copy c");
  for (int i = 0; i < n; ++i) {
    if (c[i] != a[i] + b[i]) {
      std::fprintf(stderr, "mismatch at %d\n", i);
      return EXIT_FAILURE;
    }
  }
  std::printf("OK: %d elements, %d blocks, %d threads/block\n",
              n, blocks, threadsPerBlock);

  check(cudaFree(d_a), "cudaFree a");
  check(cudaFree(d_b), "cudaFree b");
  check(cudaFree(d_c), "cudaFree c");
  return EXIT_SUCCESS;
}
