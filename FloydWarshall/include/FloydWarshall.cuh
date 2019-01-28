// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>

// #define O_TILE_WIDTH 16
// #define TILE_SIZE (O_TILE_WIDTH + 4)  // O_TILE_WIDTH + (MASK_WIDTH -1)
// #define MAX_MASK_WIDTH 10
// __constant__ float M[MAX_MASK_WIDTH * MAX_MASK_WIDTH];

#define CHECK_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
}


////////////////////////////////////////////////////////////////////////////////
//! Simple floyd_warshall kernel
//! @param d_N  input data in global memory
//! @param d_M  input mask data in global memory
//! @param d_P  output data in global memory
//! @param height  number of rows of the input matrix N
//! @param widht  number of cols of the input matrix N
////////////////////////////////////////////////////////////////////////////////
__global__
void parallel_floyd_warshall_kernel(float *N, float *P, int height, int width, const int Mask_Width);
