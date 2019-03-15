#include "FloydWarshall.hpp"
#include "FloydWarshall.cuh"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
#include <fstream>
#include <iostream>
#include <string>

// ---------------------------
#include <cuda_profiler_api.h>
// ---------------------------


using matrix_t = float;
using namespace std;

void printMatrix(float *A, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", A[i*height+j]);
        }
        printf("\n");
    }
    printf("\n");
}

void printMatrix_host(matrix_t **A, int height, int width) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            printf("%.2f ", A[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char* argv[]) {
    if (argc != 3)
        return EXIT_FAILURE;

    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);
    graph.read(argv[1]);

    ofstream cpu_times, omp_times, cuda_times;
    cpu_times.open(("cpu_times_" + to_string(graph.nV()) + ".txt").c_str(), ios::out | ios::app);
    omp_times.open(("omp_times_" + to_string(graph.nV()) + ".txt").c_str(), ios::out | ios::app);
    cuda_times.open(("cuda_times_"  + to_string(graph.nV()) + ".txt").c_str(), ios::out | ios::app);

    if (cpu_times.fail() || cpu_times.fail() || cpu_times.fail()) {
      std::cout << "Fail to create output file" << '\n';
      return EXIT_FAILURE;
    }


    auto matrix = new matrix_t*[graph.nV()];
    auto matrix_omp = new matrix_t*[graph.nV()];

    // fill with infinity
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        matrix_omp[i] = new matrix_t[graph.nV()];

        std::fill(matrix[i], matrix[i] + graph.nV(), std::numeric_limits<matrix_t>::infinity());
        std::fill(matrix_omp[i], matrix_omp[i] + graph.nV(), std::numeric_limits<matrix_t>::infinity());

    }

    // for each edge knew, insert the edge in the matrix
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
        matrix_omp[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);

    }

    // copy the matrix for the cuda parallel algorithm
    matrix_t *matrix_h , *copy_of_matrix_h;
    //matrix_t *matrix_seq_blk;
    matrix_h = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());     // input matrix
    copy_of_matrix_h = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());
    //matrix_seq_blk = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());     // input matrix
    for (int i = 0; i < graph.nV(); i++) {
      for (int j = 0; j < graph.nV(); j++) {
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(*(matrix+i)+j));
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(matrix[i]+j));

        matrix_h[i*graph.nV()+j] = matrix[i][j];
        copy_of_matrix_h[i*graph.nV()+j] = matrix[i][j];
        //matrix_seq_blk[i*graph.nV()+j] = matrix[i][j];
      }
    }

    float msTime_cuda, msTime_omp, msTime_cpu;
    cudaEvent_t startTimeCuda, stopTimeCuda;
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);


    //--------------------------------------------------------------------------
    // start SEQUENTIAL Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    cudaEventRecord(startTimeCuda, 0);
    cudaEventSynchronize(startTimeCuda);

    floyd_warshall::floyd_warshall(matrix, graph.nV());
    //floyd_warshall::floyd_warshall_opt(matrix, graph.nV());
    //floyd_warshall::floydWarshall_blk(matrix_seq_blk, graph.nV());

    cudaEventRecord(stopTimeCuda, 0);
    cudaEventSynchronize(stopTimeCuda);
    cudaEventElapsedTime(&msTime_cpu, startTimeCuda, stopTimeCuda);
    printf("HostTime CPU: %f\n", msTime_cpu);
    cpu_times << msTime_cpu << "\n";


    //--------------------------------------------------------------------------
    // start PARALLEL OMP Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    cudaEventRecord(startTimeCuda, 0);
    cudaEventSynchronize(startTimeCuda);

    floyd_warshall::floyd_warshall_omp(matrix_omp, graph.nV());

    cudaEventRecord(stopTimeCuda, 0);
    cudaEventSynchronize(stopTimeCuda);
    cudaEventElapsedTime(&msTime_omp, startTimeCuda, stopTimeCuda);
    printf("HostTime OMP: %f\n", msTime_omp);
    omp_times << msTime_omp << "\n";


    //--------------------------------------------------------------------------
    // start PARALLEL CUDA Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    for(int t = 32; t < 1025; t*=2){
      int threads_per_block = t;
      std::cout <<  t << " - " << '\n';
    for(int s = 0; s < 10;s++){
    // cudaProfilerStart();
    msTime_cuda = parallel_floyd_warshall(matrix_h, graph.nV(), atoi(argv[2]), threads_per_block);
    // cudaProfilerStop();
    cuda_times << msTime_cuda << "\n";

    // printf("Result from HOST:\n");
    // printMatrix_host(matrix, graph.nV(), graph.nV());
    // printf("\n");
    //
    // printf("Result from HOST_BLK:\n");
    // printMatrix(matrix_seq_blk, graph.nV(), graph.nV());
    // printf("\n");
    // //
    // printf("Result from GPU:\n");
    // printMatrix(matrix_h, graph.nV(), graph.nV());
    // printf("\n");


    // Verify that the result is correct
    for (int i = 0; i < graph.nV(); ++i) {
      for (int j = 0; j < graph.nV(); j++) {
        if (fabs(matrix_h[i*graph.nV()+j] - matrix[i][j]) > 1e-0) {
        //if (fabs(matrix_h[i*graph.nV()+j] - matrix_seq_blk[i*graph.nV()+j]) > 1e-2) {

            fprintf(stderr, "\033[0;31mError\033[0m: result verification failed at element [%d][%d]! -- %.2f != %.2f\n", i, j, matrix_h[i*graph.nV()+j], matrix[i][j]);
            exit(EXIT_FAILURE);
        }
      }
    }

    // Verify that the result is correct
    // for (int i = 0; i < graph.nV(); ++i) {
    //   for (int j = 0; j < graph.nV(); j++) {
    //     if (fabs(matrix_omp[i][j] - matrix[i][j]) > 1e-0) {
    //     //if (fabs(matrix_h[i*graph.nV()+j] - matrix_seq_blk[i*graph.nV()+j]) > 1e-2) {
    //
    //         fprintf(stderr, "\033[0;31mError\033[0m: result verification failed at element [%d][%d]! -- %.2f != %.2f\n", i, j, matrix_omp[i][j], matrix[i][j]);
    //         exit(EXIT_FAILURE);
    //     }
    //   }
    // }

    // SPEED UP
    //printf("Speedup CPU vs OMP: %f\n", msTime_cpu / msTime_omp);
    //printf("Speedup OMP vs GPU: %f\n", msTime_omp / msTime_cuda);
    printf("Speedup CPU vs GPU: %f\n", msTime_cpu / msTime_cuda);

    for (int i = 0; i < graph.nV(); i++) {
      for (int j = 0; j < graph.nV(); j++) {
        matrix_h[i*graph.nV()+j] = copy_of_matrix_h[i*graph.nV()+j];
      }
    }
  }
}

    // cleanup memory
    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;


    free(matrix_h);

    cpu_times.close();
    omp_times.close();
    cuda_times.close();

}
