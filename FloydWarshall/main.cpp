#include "FloydWarshall.hpp"
#include "FloydWarshall.cuh"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
// ---------------------------
#include <cuda_profiler_api.h>
// ---------------------------

using matrix_t = float;

int main(int argc, char* argv[]) {
    if (argc != 2)
        return EXIT_FAILURE;

    graph::GraphWeight<int, int, matrix_t> graph(graph::structure_prop::COO);
    graph.read(argv[1]);

    auto matrix = new matrix_t*[graph.nV()];

    // fill with infinity
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        std::fill(matrix[i], matrix[i] + graph.nV(), std::numeric_limits<matrix_t>::infinity());
    }

    // for each edge knew, insert the edge in the matrix
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
        //std::cout << matrix[std::get<0>(index)][std::get<1>(index)] << " "<< '\n';
    }

    // copy the matrix for the parallel algorithm
    matrix_t *matrix_h;
    matrix_h = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());     // input matrix
    for (int i = 0; i < graph.nV(); i++) {
      for (int j = 0; j < graph.nV(); j++) {
        printf("%.1f\t", matrix[i][j]);
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(*(matrix+i)+j));
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(matrix[i]+j));

        matrix_h[i*graph.nV()+j] = matrix[i][j];

      }
      printf("\n");
    }


    //--------------------------------------------------------------------------
    // start sequential Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    floyd_warshall::floyd_warshall(matrix, graph.nV());


    //--------------------------------------------------------------------------
    // start parallel Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    // cudaProfilerStart();
    parallel_floyd_warshall(matrix_h, graph.nV());
    // cudaProfilerStop();
    //--------------------------------------------------------------------------


    // Verify that the result matrix is correct
    for (int i = 0; i < graph.nV(); ++i) {
      for (int j = 0; j < graph.nV(); j++) {
        //printf("%.1f\t", matrix_h[i*graph.nV()+j]);
        //printf("host  [%d]  \t%f\n", i*graph.nV()+j, matrix[i][j]);
        printf("%.1f\t", matrix[i][j]);
        if (fabs(matrix_h[i*graph.nV()+j] - matrix[i][j]) > 1e-5) {
            fprintf(stderr, "Result verification failed at element [%d][%d]!\n", i, j);
            exit(EXIT_FAILURE);
        }
      }
      printf("\n");
    }

    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;

    // cleanup memory
    free(matrix_h);

}
