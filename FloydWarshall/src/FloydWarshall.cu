
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>


__global__
void parallel_floyd_warshall_kernel(float *N, float *P, int num_vertices) {


  // parallel implementation of floyd_warshall here ....

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;

  int i = bx * blockDim.x + tx;
  int j = by * blockDim.x + ty;

  if (tx == 0 &&
      ty == 0 &&
      bx == 0 &&
      by == 0) {
        printf("hello [%d][%d] -- [%d][%d]\n", ty,tx,by,bx);
  }



  //     printf("matrix[ty+by][tx+bx]  \t= %f\n", i, j, matrix[ty+by][tx+bx]);
  //     printf("matrix[%d]      = %f\n", i* graph.nV() + j, matrix[i*graph.nV()+j]);




}



// ----------------------------------------------------------------------------
// PERFORM PARALLEL FLOYD-WARSHALL
// ---------------- ------------------------------------------------------------
template <typename T>
void parallel_floyd_warshall(T* h_N, int n) {
  printf("Called parallel_floyd_warshall\n");
  const auto INF = std::numeric_limits<T>::infinity();

  float *d_N, *d_P;
  int size = n * n * sizeof(float);

  // cudaEvent_t startTimeCuda, stopTimeCuda;
  // cudaEventCreate(&startTimeCuda);
  // cudaEventCreate(&stopTimeCuda);

  // 1. Allocate global memory on the device for N, M and P
  CHECK_ERROR(cudaMalloc((void**)&d_N, size));
  CHECK_ERROR(cudaMalloc((void**)&d_P, size));

  // copy N and M to device memory
  cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

  // 2. Kernel launch code - to have the device to perform the actual convolution
  // ------------------- CUDA COMPUTATION ---------------------------
  // cudaEventRecord(startTimeCuda, 0);
  // cudaEventSynchronize(startTimeCuda);

  dim3 dimGrid(ceil(n / 4.0), ceil(n / 4.0), 1);
  dim3 dimBlock(4.0, 4.0, 1.0);
  parallel_floyd_warshall_kernel <<< dimGrid, dimBlock >>> (d_N, d_P, n);

  // cudaEventRecord(stopTimeCuda, 0);
  // cudaEventSynchronize(stopTimeCuda);

  // ---------------------- CUDA ENDING -----------------------------
  // float msTime;
  // cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
  // printf("KernelTime: %fn", msTime);

  // 3. copy C from the device memory
  cudaMemcpy(h_N, d_P, size, cudaMemcpyDeviceToHost);

  // // cleanup memory
  cudaFree(d_N);
  cudaFree(d_P);

  printf("return from parallel_floyd_warshall\n");
  // return msTime;
}

template void parallel_floyd_warshall<float>(float*, int);
