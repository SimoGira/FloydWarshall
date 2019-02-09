
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>

#define BLOCK_SIZE 8

__constant__ auto INF = std::numeric_limits<float>::infinity();   // qui andrebbe sistemato in modo che al posto di float accetti T

__global__
void parallel_floyd_warshall_kernel(float *N, int n, int k) {

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int row = by * blockDim.y + ty;
  int col = bx * blockDim.x + tx;

  // check for a valid range
  if (row != col && row < n && col < n) {

      float r_k_value = N[row * n + k];
      float k_c_value = N[k * n + col];
      float r_c_value = N[row * n + col];

      if (r_k_value != INF && k_c_value != INF) {
          float sum = r_k_value + k_c_value;
          if (sum < r_c_value) {
              N[row * n + col] = sum;
          }
            // TODO Da sistemare ...
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
