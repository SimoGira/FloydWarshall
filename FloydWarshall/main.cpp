#include "FloydWarshall.hpp"
#include "FloydWarshall.cuh"
#include "Graph/GraphWeight.hpp"
#include <tuple>
#include <limits>
// ---------------------------
#include <cuda_profiler_api.h>
// ---------------------------









#define BLOCK_DIM 32
#include <algorithm>
void floydWarshall_seq(float *matrix, int nV) {
    int numBlock = (nV-1)/BLOCK_DIM + 1;
    float *temp = new float[nV * nV];
    for(int bId=0; bId < numBlock; bId++) {
        //fase 1
        for(int k=bId * BLOCK_DIM;k<(bId+1)*BLOCK_DIM;k++) {
            //memcpy(temp, matrix, nV * nV * sizeof(float));
            for(int i=bId*BLOCK_DIM;i<(bId+1)*BLOCK_DIM;i++) {
                for(int j=bId*BLOCK_DIM;j<(bId+1)*BLOCK_DIM;j++) {
                    if(i < nV && j < nV && k < nV) {
                        matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                    }
                }
            }
            //memcpy(matrix, temp, nV * nV * sizeof(float));
        }

        // fase 2
        // blocchi i-allineati
        for(int ib=0;ib<numBlock;ib++) {
            //per ogni blocco
            if(ib != bId) {
                for(int k=bId * BLOCK_DIM;k<(bId+1)*BLOCK_DIM;k++) {
                    //memcpy(temp, matrix, nV * nV * sizeof(float));
                    for(int i=bId*BLOCK_DIM;i<(bId+1)*BLOCK_DIM;i++) {
                        for(int j=ib*BLOCK_DIM;j<(ib+1)*BLOCK_DIM;j++) {
                            if(i < nV && j < nV && k < nV) {
                                matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                            }
                        }
                    }
                    //memcpy(matrix, temp, nV * nV * sizeof(float));
                }
            }
        }
        // //blocchi j-allineati
        for(int jb=0;jb<numBlock;jb++) {
            //per ogni blocco
            if(jb != bId) {
                for(int k=bId * BLOCK_DIM;k<(bId+1)*BLOCK_DIM;k++) {
                    //memcpy(temp, matrix, nV * nV * sizeof(float));
                    for(int i=jb*BLOCK_DIM;i<(jb+1)*BLOCK_DIM;i++) {
                        for(int j=bId*BLOCK_DIM;j<(bId+1)*BLOCK_DIM;j++) {
                            if(i < nV && j < nV && k < nV) {
                                matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                            }
                        }
                    }
                    //memcpy(matrix, temp, nV * nV * sizeof(float));
                }
            }
        }

        //fase 3
        for(int ib=0;ib<numBlock;ib++) {
            for(int jb=0;jb<numBlock;jb++) {
                //per ogni blocco
                if(ib != bId && jb != bId) {
                    for(int k=bId * BLOCK_DIM;k<(bId+1)*BLOCK_DIM;k++) {
                        //memcpy(temp, matrix, nV * nV * sizeof(float));
                        for(int i=jb*BLOCK_DIM;i<(jb+1)*BLOCK_DIM;i++) {
                            for(int j=ib*BLOCK_DIM;j<(ib+1)*BLOCK_DIM;j++) {
                                if(i < nV && j < nV && k < nV) {
                                    matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                                }
                            }
                        }
                        //memcpy(matrix, temp, nV * nV * sizeof(float));
                    }
                }
            }
        }
        //break;
    }
    delete[] temp;
}
























using matrix_t = float;

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
    }

    // copy the matrix for the parallel algorithm
    matrix_t *matrix_h;
    //matrix_t *matrix_seq_blk;
    matrix_h = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());     // input matrix
    //matrix_seq_blk = (float*)malloc(sizeof(float)*graph.nV()*graph.nV());     // input matrix
    for (int i = 0; i < graph.nV(); i++) {
      for (int j = 0; j < graph.nV(); j++) {
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(*(matrix+i)+j));
        //printf("matrix[%d]      = %f\n", i* graph.nV() + j, *(matrix[i]+j));

        matrix_h[i*graph.nV()+j] = matrix[i][j];
        //matrix_seq_blk[i*graph.nV()+j] = matrix[i][j];
      }
    }

    float msTime, msTime_seq;
    cudaEvent_t startTimeCuda, stopTimeCuda;
    cudaEventCreate(&startTimeCuda);
    cudaEventCreate(&stopTimeCuda);


    //--------------------------------------------------------------------------
    // start sequential Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    cudaEventRecord(startTimeCuda, 0);
    cudaEventSynchronize(startTimeCuda);

    floyd_warshall::floyd_warshall(matrix, graph.nV());
    //floydWarshall_seq(matrix_seq_blk, graph.nV());

    cudaEventRecord(stopTimeCuda, 0);
    cudaEventSynchronize(stopTimeCuda);
    cudaEventElapsedTime(&msTime_seq, startTimeCuda, stopTimeCuda);
    printf("HostTime: %f\n", msTime_seq);


    //--------------------------------------------------------------------------
    // start parallel Floyd Warshall algorithm
    //--------------------------------------------------------------------------
    // cudaProfilerStart();
    msTime = parallel_floyd_warshall(matrix_h, graph.nV(), atoi(argv[2]));
    // cudaProfilerStop();


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

    // SPEED UP
    printf("Speedup: %f\n", msTime_seq / msTime);

    // cleanup memory
    for (int i = 0; i < graph.nV(); i++)
        delete[] matrix[i];
    delete[] matrix;


    free(matrix_h);

}
