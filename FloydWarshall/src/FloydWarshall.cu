
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>

#define BLOCK_SIZE 4

__constant__ auto INF = std::numeric_limits<float>::infinity();   // qui andrebbe sistemato in modo che al posto di float accetti T

__global__
void parallel_floyd_warshall_kernel(float *N, int num_vertices) {

  // Block index
  int bx = blockIdx.x;
  int by = blockIdx.y;

  // Thread index
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  int i = bx * blockDim.x + tx;
  int j = by * blockDim.y + ty;

  // check for a valid range
  if (i < num_vertices && j < num_vertices){

    for (int k = 0; k < num_vertices; k++) {
      if (N[i * num_vertices + k] != INF &&
          N[k * num_vertices + j] != INF &&
          N[i * num_vertices + k] + N[k * num_vertices + j] < N[i * num_vertices +j]) {

            N[i * num_vertices + j] = N[i * num_vertices + k] + N[k * num_vertices + j];

            // TODO Da sistemare ...
      }
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
  parallel_floyd_warshall_kernel <<< dimGrid, dimBlock >>> (d_N, n);

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

  printf("return from parallel_floyd_warshall\n");
  // return msTime;
}

template void parallel_floyd_warshall<float>(float*, int);
