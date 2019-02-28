
#pragma once
#include "FloydWarshall.cuh"
#include "Kernels.cuh"

// ----------------------------------------------------------------------------
// PERFORM PARALLEL FLOYD-WARSHALL
// ----------------------------------------------------------------------------
template <typename T>
void parallel_floyd_warshall(T* h_N, int n, int kernel_number) {
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

  dim3 dimGrid(ceil(n / (float)BLOCK_SIZE), ceil(n / (float)BLOCK_SIZE), 1);
  dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE, 1.0);

  printf("Grid:   {%d,\t%d,\t%d} blocks.\nBlocks: {%d,\t%d,\t%d} threads.\n", \
          dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  // 2. Kernel launch code - to have the device to perform the Floyd Warshall algorithm
  // ------------------- CUDA COMPUTATION ---------------------------
  // cudaEventRecord(startTimeCuda, 0);
  // cudaEventSynchronize(startTimeCuda);

  switch (kernel_number) {
    case 1:
      for (int k = 0; k < n; k++) {
        naive_floyd_warshall_kernel <<< dimGrid, dimBlock >>> (d_N, n, k);
      }
      break;
    case 2:
      dim3 dimGrid(n, n, 1);                      // < ---- check!!!
      for(int k = 0; k < n; ++k)
        coa_floyd_warshall_kernel<<<dimGrid, 1>>>(d_N, n, k);
      break;
    case 3:
      dim3 dimGrid(n, n / TILE_WIDTH, 1);
      dim3 dimBlock(1, TILE_WIDTH, 1);
      for(int k = 0; k < n; ++k)
        sm_floyd_warshall_kernel<<<dimGrid, dimBlock>>>(d_N, n, k);
      break;
    case 4:
      // TODO Blocked_kernel
      break;
    default:
      break;
  }

  // cudaEventRecord(stopTimeCuda, 0);
  // cudaEventSynchronize(stopTimeCuda);

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel %d (error code %s)!\n", kernel_number, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

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

template void parallel_floyd_warshall<float>(float*, int, int);
