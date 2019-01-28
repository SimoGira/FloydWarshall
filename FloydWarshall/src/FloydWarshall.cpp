#include "FloydWarshall.hpp"
#include <iostream>
#include <limits>

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
void parallel_floyd_warshall(T** matrix, int num_vertices) {
    const auto INF = std::numeric_limits<T>::infinity();


    // prepare for parallel_floyd_warshall_kernel here ....


}



template void parallel_floyd_warshall<float>(float**, int);

} // floyd_warshall
