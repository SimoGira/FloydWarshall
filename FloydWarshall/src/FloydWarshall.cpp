#include "FloydWarshall.hpp"
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>



#define CHECK_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(err); \
    } \
}


namespace floyd_warshall {

template<typename T>
void floyd_warshall(T** matrix, int num_vertices) {
    const auto INF = std::numeric_limits<T>::infinity();

    for (int k = 0; k < num_vertices; k++) {
        for (int i = 0; i < num_vertices; i++) {
            for (int j = 0; j < num_vertices; j++) {
                if (matrix[i][k] != INF &&
                    matrix[k][j] != INF &&
                    matrix[i][k] + matrix[k][j] < matrix[i][j]) {

                    matrix[i][j] = matrix[i][k] + matrix[k][j];
                }
            }
        }
    }
}

template void floyd_warshall<float>(float**, int);


// ----------------------------------------------------------------------------
// PERFORM PARALLEL FLOYD-WARSHALL
// ---------------- ------------------------------------------------------------
template<typename T>
void parallel_floyd_warshall(T* h_N, int n) {
    const auto INF = std::numeric_limits<T>::infinity();

    float *d_N, *d_P;
    int size = n * n * sizeof(float);

    // cudaEvent_t startTimeCuda, stopTimeCuda;
    // cudaEventCreate(&startTimeCuda);
    // cudaEventCreate(&stopTimeCuda);

    //1. Allocate global memory on the device for N, M and P
    CHECK_ERROR(cudaMalloc((void**)&d_N, size));
    CHECK_ERROR(cudaMalloc((void**)&d_P, size));

    // copy N and M to device memory
    cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);

    //2. Kernel launch code - to have the device to perform the actual convolution
    // ------------------- CUDA COMPUTATION ---------------------------
    // cudaEventRecord(startTimeCuda, 0);
	// cudaEventSynchronize(startTimeCuda);

    dim3 dimGrid(ceil(n / 4.0), ceil(n / 4.0), 1);
    dim3 dimBlock(4.0, 4.0, 1);
    parallel_floyd_warshall_kernel << <dimGrid, dimBlock >> >(d_N, d_P, n);

    // cudaEventRecord(stopTimeCuda, 0);
	// cudaEventSynchronize(stopTimeCuda);

    // ---------------------- CUDA ENDING -----------------------------
    // float msTime;
    // cudaEventElapsedTime(&msTime, startTimeCuda, stopTimeCuda);
    // printf("KernelTime: %f\n", msTime);

    //3. copy C from the device memory
    cudaMemcpy(h_N, d_P, size, cudaMemcpyDeviceToHost);

    // // cleanup memory
    cudaFree(d_N);
    cudaFree(d_P);

    //return msTime;


    // prepare for parallel_floyd_warshall_kernel here ....


}



template void parallel_floyd_warshall<float>(float*, int);

} // floyd_warshall
