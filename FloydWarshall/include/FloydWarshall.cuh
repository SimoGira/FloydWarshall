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


template<typename T>
void parallel_floyd_warshall(T* h_N, int n, int kernel_number);
