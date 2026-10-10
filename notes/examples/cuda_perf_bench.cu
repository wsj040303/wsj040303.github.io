#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t result, const char* step) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", step, cudaGetErrorString(result));
    std::exit(EXIT_FAILURE);
  }
}

// 两个版本计算相同的 c[i] = a[i] + b[i]，只改变线程到元素的映射。
__global__ void coalesced_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

__global__ void strided_add(const float* a, const float* b, float* c, int n) {
  int i = threadIdx.x * gridDim.x + blockIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

using Kernel = void (*)(const float*, const float*, float*, int);

static void launch(Kernel kernel, const float* a, const float* b, float* c,
                   int n, int blocks, int threads) {
  kernel<<<blocks, threads>>>(a, b, c, n);
  check(cudaGetLastError(), "kernel launch");
}

static void verify(Kernel kernel, const float* a, const float* b, float* c,
                   int n, int blocks, int threads, std::vector<float>& output) {
  launch(kernel, a, b, c, n, blocks, threads);
  check(cudaDeviceSynchronize(), "kernel execution");
  check(cudaMemcpy(output.data(), c, output.size() * sizeof(float),
                   cudaMemcpyDeviceToHost), "copy result");
  for (int i = 0; i < n; ++i) {
    if (output[i] != 2.0f) {
      std::fprintf(stderr, "mismatch at %d: %.8g\n", i, output[i]);
      std::exit(EXIT_FAILURE);
    }
  }
}

static float median_us(Kernel kernel, const float* a, const float* b, float* c,
                       int n, int blocks, int threads) {
  constexpr int warmup = 5;
  constexpr int repeat = 20;
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "create start event");
  check(cudaEventCreate(&stop), "create stop event");
  for (int i = 0; i < warmup; ++i) launch(kernel, a, b, c, n, blocks, threads);
  check(cudaDeviceSynchronize(), "warmup");

  std::vector<float> times;
  times.reserve(repeat);
  for (int i = 0; i < repeat; ++i) {
    check(cudaEventRecord(start), "record start");
    launch(kernel, a, b, c, n, blocks, threads);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "synchronize stop");
    float milliseconds = 0.0f;
    check(cudaEventElapsedTime(&milliseconds, start, stop), "elapsed time");
    times.push_back(milliseconds * 1000.0f);
  }
  std::sort(times.begin(), times.end());
  check(cudaEventDestroy(start), "destroy start event");
  check(cudaEventDestroy(stop), "destroy stop event");
  return (times[repeat / 2 - 1] + times[repeat / 2]) / 2.0f;
}

int main() {
  constexpr int n = 1 << 24;
  constexpr int threads = 256;
  constexpr int blocks = (n + threads - 1) / threads;
  const size_t bytes = static_cast<size_t>(n) * sizeof(float);
  std::vector<float> input(n, 1.0f), output(n);
  float *a = nullptr, *b = nullptr, *c = nullptr;
  check(cudaMalloc(&a, bytes), "allocate a");
  check(cudaMalloc(&b, bytes), "allocate b");
  check(cudaMalloc(&c, bytes), "allocate c");
  check(cudaMemcpy(a, input.data(), bytes, cudaMemcpyHostToDevice), "copy a");
  check(cudaMemcpy(b, input.data(), bytes, cudaMemcpyHostToDevice), "copy b");

  verify(coalesced_add, a, b, c, n, blocks, threads, output);
  verify(strided_add, a, b, c, n, blocks, threads, output);
  const float contiguous_us = median_us(coalesced_add, a, b, c, n, blocks, threads);
  const float strided_us = median_us(strided_add, a, b, c, n, blocks, threads);
  const double gb = 3.0 * bytes / 1e9;
  std::printf("verified %d elements, block=%d, 5 warmup + 20 samples per kernel\n",
              n, threads);
  std::printf("coalesced: median %.1f us, effective %.1f GB/s\n",
              contiguous_us, gb / (contiguous_us * 1e-6));
  std::printf("strided:   median %.1f us, effective %.1f GB/s\n",
              strided_us, gb / (strided_us * 1e-6));

  check(cudaFree(a), "free a");
  check(cudaFree(b), "free b");
  check(cudaFree(c), "free c");
}
