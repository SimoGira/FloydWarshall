
#include "FloydWarshall.cuh"
#include "Kernels.cuh"

// ----------------------------------------------------------------------------
// PERFORM PARALLEL FLOYD-WARSHALL
// ----------------------------------------------------------------------------
template <typename T>
float parallel_floyd_warshall(T* h_N, int n, int kernel_number, int threads_per_block) {
  float *d_N;
  int size = n * n * sizeof(float);

  cudaEvent_t startTimeCuda, stopTimeCuda;
  cudaEventCreate(&startTimeCuda);
  cudaEventCreate(&stopTimeCuda);

  // 1. Allocate global memory on the device for N
  CHECK_ERROR(cudaMalloc((void**)&d_N, size));

  // copy N to device memory
  CHECK_ERROR(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));

  dim3 dimGrid;
  dim3 dimBlock;

// For blocked algorithm
/******************************************************************************/
  int stages = ceil(n / (float)TILE_WIDTH);

  // dimensions
  dim3 blockSize(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 phase1Grid(1, 1, 1);
  dim3 phase2Grid(stages-1, 2, 1);
  dim3 phase3Grid(stages, stages, 1);
  //dim3 phase3Grid(stages-1, stages-1, 1);
/******************************************************************************/

  // printf("Grid:   {%d,\t%d,\t%d} blocks.\nBlocks: {%d,\t%d,\t%d} threads.\n", \
  //         dimGrid.x, dimGrid.y, dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z);

  // 2. Kernel launch code - to have the device to perform the Floyd Warshall algorithm
  // ------------------- CUDA COMPUTATION ---------------------------
  cudaEventRecord(startTimeCuda, 0);
  cudaEventSynchronize(startTimeCuda);

  switch (kernel_number) {
    case 1:
      dimGrid  = dim3( ceil(n / (float)BLOCK_NAIVE), ceil(n / (float)BLOCK_NAIVE), 1.0);
      dimBlock = dim3(BLOCK_NAIVE, BLOCK_NAIVE,1.0);

      for (int k = 0; k < n; k++)
        naive_floyd_warshall_kernel <<< dimGrid, dimBlock >>> (d_N, n, k);
      break;
    case 2:
      //dimGrid = dim3( ceil(n / (float)BLOCK_SIZE), ceil(n / (float)BLOCK_SIZE), 1.0);
      dimGrid  = dim3(ceil((float)n*n/(BLOCK_COA*SEGMENT_SIZE)));
      dimBlock = dim3(BLOCK_COA, 1.0,1.0);

      for(int k = 0; k < n; ++k)
        coa_floyd_warshall_kernel<<<dimGrid, dimBlock>>>(d_N, n, k);
      break;
    case 3:
      dimGrid  = dim3(ceil(n / (float)BLOCK_SM), n, 1.0);
      dimBlock = dim3(BLOCK_SM, 1.0, 1.0);

      for(int k = 0; k < n; ++k)
        sm_floyd_warshall_kernel<<<dimGrid, dimBlock>>>(d_N, n, k);
      break;
    case 4:

      for(int k = 0; k < stages; k++) {
    		int base = TILE_WIDTH * k;
        phase1<<<phase1Grid, blockSize>>>(d_N, n, base);
        phase2<<<phase2Grid, blockSize>>>(d_N, n, k, base);
        phase3<<<phase3Grid, blockSize>>>(d_N, n, k, base);
      }

      break;
    default:
      break;
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to launch kernel %d (error code %s)!\n", kernel_number, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEventRecord(stopTimeCuda, 0);
  cudaEventSynchronize(stopTimeCuda);


  // ---------------------- CUDA ENDING -----------------------------
  float msTime;
  cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
  printf("DeviceTime: %f\n\n", msTime);

  // 3. copy result from the device memory
  CHECK_ERROR(cudaMemcpy(h_N, d_N, size, cudaMemcpyDeviceToHost));

  // cleanup memory
  CHECK_ERROR(cudaFree(d_N));

  return msTime;
}

template float parallel_floyd_warshall<float>(float*, int, int, int);
