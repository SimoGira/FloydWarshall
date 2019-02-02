
#include "FloydWarshall.cuh"


__global__
void parallel_floyd_warshall_kernel(float *N, float *P, int num_vertices) {


  // parallel implementation of floyd_warshall here ....

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.y;
  int by = blockIdx.x;

  //     printf("matrix[ty+by][tx+bx]  \t= %f\n", i, j, matrix[ty+by][tx+bx]);
  //     printf("matrix[%d]      = %f\n", i* graph.nV() + j, matrix[i*graph.nV()+j]);




}
