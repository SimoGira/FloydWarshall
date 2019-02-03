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
    for (int i = 0; i < graph.nV(); i++) {
        matrix[i] = new matrix_t[graph.nV()];
        std::fill(matrix[i], matrix[i] + graph.nV(),
                  std::numeric_limits<matrix_t>::infinity());
    }
    for (int i = 0; i < graph.nE(); i++) {
        auto index = graph.coo_ptr()[i];
        matrix[std::get<0>(index)][std::get<1>(index)] = std::get<2>(index);
    }

    floyd_warshall::floyd_warshall(matrix, graph.nV());

    //--------------------------------------------------------------------------
    matrix_t *matrix_h;
    matrix_h = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());     // input matrix

    for (int i = 0; i < graph.nV(); i++) {
      for (int j = 0; j < graph.nV(); j++) {
        //printf("%f ", i, j, matrix[i][j]);
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(*(matrix+i)+j));
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(matrix[i]+j));

        matrix_h[i*graph.nV()+j] = matrix[i][j];
        //printf("%f ", i, j, matrix_h[i*graph.nV()+j]);

        // DONE mi faccio la mia matrice salvandomi i valori da questa oppure devo trovare un altro modo per referenziare questa

      }
      //printf("\n");
    }



    // cudaProfilerStart();
    //
    parallel_floyd_warshall(matrix_h, graph.nV());
    //
    // cudaProfilerStop();
    //--------------------------------------------------------------------------

    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;

    // cleanup memory
    free(matrix_h);

}
