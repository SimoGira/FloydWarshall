// includes CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda.h>


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
void parallel_floyd_warshall_kernel(float *N, int num_vertices);


template<typename T>
void parallel_floyd_warshall(T* h_N, int n);
