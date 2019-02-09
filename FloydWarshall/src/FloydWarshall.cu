
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>

#define BLOCK_SIZE 10

__constant__ auto INF = std::numeric_limits<float>::infinity();   // qui andrebbe sistemato in modo che al posto di float accetti T

__global__
void parallel_floyd_warshall_kernel(float *N, int n, int k) {

  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  // check for a valid range
  if (i >= n || j >= n || k >= n || i == j) return;

  const float i_k_value = N[i * n + k];
  const float k_j_value = N[k * n + j];
  const float i_j_value = N[i * n + j];

  if (i_k_value != INF && k_j_value != INF) {
      float sum = i_k_value + k_j_value;
      if (sum < i_j_value) {
          N[i * n + j] = sum;
      }
  }
}



// ----------------------------------------------------------------------------
// PERFORM PARALLEL FLOYD-WARSHALL
// ---------------- ------------------------------------------------------------
template <typename T>
void parallel_floyd_warshall(T* h_N, int n) {
  printf("Called parallel_floyd_warshall\n");

  float *d_N;
  int size = n * n * sizeof(float);

  // cudaEvent_t startTimeCuda, stopTimeCuda;
  // cudaEventCreate(&startTimeCuda);
  // cudaEventCreate(&stopTimeCuda);

  // 1. Allocate global memory on the device for N
  CHECK_ERROR(cudaMalloc((void**)&d_N, size));

  // copy N to device memory
  CHECK_ERROR(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));

  // copy infinty constant to constant memory
  //CHECK_ERROR(cudaMemcpyToSymbol(dest, source, size));

  // 2. Kernel launch code - to have the device to perform the Floyd Warshall algorithm
  // ------------------- CUDA COMPUTATION ---------------------------
  // cudaEventRecord(startTimeCuda, 0);
  // cudaEventSynchronize(startTimeCuda);

  dim3 dimGrid(ceil(n / BLOCK_SIZE), ceil(n / BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1.0);

  for (int k = 0; k < n; k++) {
    parallel_floyd_warshall_kernel <<< dimGrid, dimBlock >>> (d_N, n, k);
  }

  // cudaEventRecord(stopTimeCuda, 0);
  // cudaEventSynchronize(stopTimeCuda);

  // ---------------------- CUDA ENDING -----------------------------
  // float msTime;
  // cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
  // printf("KernelTime: %fn", msTime);

  // 3. copy result from the device memory
  CHECK_ERROR(cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost));

  // // cleanup memory
  CHECK_ERROR(cudaFree(d_N));

  printf("return from parallel_floyd_warshall\n\n");
  // return msTime;
}

template void parallel_floyd_warshall<float>(float*, int);
