#include "FloydWarshall.hpp"
#include <iostream>
#include <limits>
#include <algorithm>
#include <omp.h>
#define BLOCK_DIM_SEQ 32

namespace floyd_warshall {

  //////////////////////////////////////////////////////////////////////////////
  // ORIGINAL
  //////////////////////////////////////////////////////////////////////////////
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


  //////////////////////////////////////////////////////////////////////////////
  // OPTIMIZED
  //////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void floyd_warshall_opt(T** matrix, int num_vertices) {
    const auto INF = std::numeric_limits<T>::infinity();

    for (int k = 0; k < num_vertices; k++) {
      for (int i = 0; i < num_vertices; i++) {
          if (matrix[i][k] != INF) {
              for (int j = 0; j < num_vertices; j++) {
                if (matrix[k][j] != INF &&
                    matrix[i][k] + matrix[k][j] < matrix[i][j]) {
                      matrix[i][j] = matrix[i][k] + matrix[k][j];
                }
              }
          }
      }
    }
  }
  template void floyd_warshall_opt<float>(float**, int);


  //////////////////////////////////////////////////////////////////////////////
  // BLOCKED
  //////////////////////////////////////////////////////////////////////////////
  template <typename T>
  void floydWarshall_blk(T* matrix, int nV) {
      int numBlock = (nV-1)/BLOCK_DIM_SEQ + 1;
      for(int bId=0; bId < numBlock; bId++) {
          //phase 1
          for(int k=bId * BLOCK_DIM_SEQ;k<(bId+1)*BLOCK_DIM_SEQ;k++) {
              for(int i=bId*BLOCK_DIM_SEQ;i<(bId+1)*BLOCK_DIM_SEQ;i++) {
                  for(int j=bId*BLOCK_DIM_SEQ;j<(bId+1)*BLOCK_DIM_SEQ;j++) {
                      if(i < nV && j < nV && k < nV) {
                          matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                      }
                  }
              }
          }

          // phase 2
          // rows-aligned blocks
          for(int ib=0;ib<numBlock;ib++) {
              // for each block
              if(ib != bId) {
                  for(int k=bId * BLOCK_DIM_SEQ;k<(bId+1)*BLOCK_DIM_SEQ;k++) {
                      for(int i=bId*BLOCK_DIM_SEQ;i<(bId+1)*BLOCK_DIM_SEQ;i++) {
                          for(int j=ib*BLOCK_DIM_SEQ;j<(ib+1)*BLOCK_DIM_SEQ;j++) {
                              if(i < nV && j < nV && k < nV) {
                                  matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                              }
                          }
                      }
                  }
              }
          }
          // cols-aligned blocks
          for(int jb=0;jb<numBlock;jb++) {
              // for each block
              if(jb != bId) {
                  for(int k=bId * BLOCK_DIM_SEQ;k<(bId+1)*BLOCK_DIM_SEQ;k++) {
                      for(int i=jb*BLOCK_DIM_SEQ;i<(jb+1)*BLOCK_DIM_SEQ;i++) {
                          for(int j=bId*BLOCK_DIM_SEQ;j<(bId+1)*BLOCK_DIM_SEQ;j++) {
                              if(i < nV && j < nV && k < nV) {
                                  matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                              }
                          }
                      }
                  }
              }
          }

          // phase 3
          for(int ib=0;ib<numBlock;ib++) {
              for(int jb=0;jb<numBlock;jb++) {
                  // for each block
                  if(ib != bId && jb != bId) {
                      for(int k=bId * BLOCK_DIM_SEQ;k<(bId+1)*BLOCK_DIM_SEQ;k++) {
                          for(int i=jb*BLOCK_DIM_SEQ;i<(jb+1)*BLOCK_DIM_SEQ;i++) {
                              for(int j=ib*BLOCK_DIM_SEQ;j<(ib+1)*BLOCK_DIM_SEQ;j++) {
                                  if(i < nV && j < nV && k < nV) {
                                      matrix[i*nV + j] = std::min(matrix[i*nV + j], matrix[i*nV + k] + matrix[k*nV + j]);
                                  }
                              }
                          }
                      }
                  }
              }
          }
          //break;
      }
  }
  template void floydWarshall_blk<float>(float*, int);


  //////////////////////////////////////////////////////////////////////////////
  // OMP
  //////////////////////////////////////////////////////////////////////////////
  template<typename T>
  void floyd_warshall_omp(T** matrix, int num_vertices) {
      const auto INF = std::numeric_limits<T>::infinity();
      int i,j,k;

      #pragma omp parallel shared(num_vertices, matrix) private(i,j, k) default(none)
      for (k = 0; k < num_vertices; k++) {

          #pragma omp for schedule(dynamic)
          for (i = 0; i < num_vertices; i++) {
              if (matrix[i][k] != INF) {
                for (j = 0; j < num_vertices; j++) {
                    if (matrix[k][j] != INF && matrix[i][k] + matrix[k][j] < matrix[i][j]) {
                        matrix[i][j] = matrix[i][k] + matrix[k][j];
                    }
                }
              }
          }
      }
  }
  template void floyd_warshall_omp<float>(float**, int);



} // floyd_warshall
