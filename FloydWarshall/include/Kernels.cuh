#pragma once
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>

#define BLOCK_SIZE 32
#define TILE_WIDTH 32

__constant__ auto INF = std::numeric_limits<float>::infinity();   // qui andrebbe sistemato in modo che al posto di float accetti T

////////////////////////////////////////////////////////////////////////////////
//! Naive floyd_warshall kernel implementation
//! @param d_N  input data in global memory
//! @param n  number of verticies of the input matrix N
//! @param k  index of the intermediate vertex
////////////////////////////////////////////////////////////////////////////////
__global__ void naive_floyd_warshall_kernel(float *N, int n, int k) {
    const unsigned int i = blockIdx.y;
    const unsigned int j = blockIdx.x;

    // check for a valid range
    if (i >= n || j >= n || k >= n) return;

    const float i_k_value = N[i * n + k];
    const float k_j_value = N[k * n + j];
    const float i_j_value = N[i * n + j];

    // calculate shortest path
    if (i_k_value != INF && k_j_value != INF) {
        float sum = i_k_value + k_j_value;
        if (sum < i_j_value) {
            N[i * n + j] = sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Coalized floyd_warshall kernel implementation
//! @param d_N  input data in global memory
//! @param n  number of verticies of the input matrix N
//! @param k  index of the intermediate vertex
////////////////////////////////////////////////////////////////////////////////
__global__ void coa_floyd_warshall_kernel(float *N, int n, int k) {

    const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    // check for a valid range
    if (i >= n || j >= n || k >= n) return;

    const float i_k_value = N[i * n + k];
    const float k_j_value = N[k * n + j];
    const float i_j_value = N[i * n + j];

    // calculate shortest path
    if (i_k_value != INF && k_j_value != INF) {
        float sum = i_k_value + k_j_value;
        if (sum < i_j_value) {
            N[i * n + j] = sum;
        }
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Shared Memory floyd_warshall kernel implementation
//! @param d_N  input data in global memory
//! @param n  number of verticies of the input matrix N
//! @param k  index of the intermediate vertex
//! @brief Here there is not warp divergence but it's missing memory coalescing
////////////////////////////////////////////////////////////////////////////////
__global__ void sm_floyd_warshall_kernel(float *N, int n, int k) {

  const unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

  // check for a valid range
  if (i >= n || j >= n || k >= n) return;

  // read in dependent values
  float i_j_value = N[i * n + j];
  float k_j_value = N[k * n + j];

  __shared__ float i_k_value;

  if (threadIdx.x == 0) {
    i_k_value = N[i * n + k];
  }
  __syncthreads();

  //printf("k = %d: t[%d][%d],\tb[%d][%d] -- i_k_value = %.1f\n",k, threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, i_k_value);

  // calculate shortest path
  if(i_k_value != INF && k_j_value != INF) {
    float sum = i_k_value + k_j_value;
    if (sum < i_j_value) {
      N[i * n + j] = sum;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//! 3-Phase parallel blocked floyd_warshall kernel implementation
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//! This kernel computes the first phase (self-dependent block)
//! @param matrix A pointer to the adjacency matrix
//! @param size   The width of the matrix
//! @param base   The base index for a block
////////////////////////////////////////////////////////////////////////////////
__global__ void phase1(float *matrix, int size, int base) {

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int i = base + ty;
  int j = base + tx;

  // computes the index for a thread
  int index = i * size + j;

  // loads data from global memory to shared memory
  __shared__ float subMatrix[TILE_WIDTH][TILE_WIDTH];
  subMatrix[ty][tx] = (i < size && j < size) ? matrix[index] : INF;
  __syncthreads();


  if (i >= size || j >= size) return;


  // run Floyd-Warshall
  float sum;
  for (int k = 0; k < TILE_WIDTH; k++) {
    sum = subMatrix[ty][k] + subMatrix[k][tx];
    if (sum < subMatrix[ty][tx]) {
      subMatrix[ty][tx] = sum;
    }
    __syncthreads();
  }

  // write back to global memory
  matrix[index] = subMatrix[ty][tx];
}

////////////////////////////////////////////////////////////////////////////////
//! This kernel computes the second phase (singly-dependent blocks)
//! @param matrix A pointer to the adjacency matrix
//! @param size   The width of the matrix
//! @param stage  The current stage of the algorithm
//! @param base   The base index for a block
////////////////////////////////////////////////////////////////////////////////
__global__ void phase2(float *matrix, int size, int stage, int base) {

  int ty = threadIdx.y;
  int tx = threadIdx.x;

  int by = blockIdx.y;
  int bx = blockIdx.x;

  // primary matrix is the matrix of the pivot (computed in phase 1)

  int i, j, i_prim, j_prim;
  i_prim = base + ty;  // pivot rows
  j_prim = base + tx;  // pivot cols


  // here we have only 2 rows in the grid, then blockIdx.y can be only 0 or 1
  if (by) { // load for column
    i = (bx < stage) ? TILE_WIDTH * bx + ty : TILE_WIDTH * (bx + 1) + ty;
    j = j_prim;
  } else {  // load for row
    i = i_prim;
    j = (bx < stage) ?  TILE_WIDTH * bx + tx : TILE_WIDTH * (bx + 1) + tx;
  }

  int index = i * size + j;
  int index_prim = i_prim * size + j_prim;


  // loads data from global memory to shared memory
  __shared__ float ownMatrix[TILE_WIDTH][TILE_WIDTH];
  __shared__ float primaryMatrix[TILE_WIDTH][TILE_WIDTH];

  ownMatrix[ty][tx] =  (i < size && j < size) ? matrix[index] : INF;
  primaryMatrix[ty][tx] = (i_prim < size && j_prim < size) ? matrix[index_prim] : INF;
  __syncthreads();


  if (i >= size || j >= size) return;


  // run Floyd Warshall
  float sum;
  if (by) {
    for (int k = 0; k < TILE_WIDTH; k++) {
      sum = ownMatrix[ty][k] + primaryMatrix[k][tx];
      if (sum < ownMatrix[ty][tx]) {
          ownMatrix[ty][tx] = sum;
      }
      __syncthreads();
    }
  }
  else {
    for (int k = 0; k < TILE_WIDTH; k++) {
      sum = primaryMatrix[ty][k] + ownMatrix[k][tx];
      if (sum < ownMatrix[ty][tx]) {
          ownMatrix[ty][tx] = sum;
      }
      __syncthreads();
    }
  }

  // write back to global memory
  matrix[index] = ownMatrix[ty][tx];
}


////////////////////////////////////////////////////////////////////////////////
 //! This kernel computes the third phase (doubly-dependent blocks)
 //! @param matrix A pointer to the adjacency matrix
 //! @param size   The width of the matrix
 //! @param stage  The current stage of the algorithm
 //! @param base   The base index for a block
 ////////////////////////////////////////////////////////////////////////////////
 __global__ void phase3(float *matrix, int size, int stage, int base) {

   int tx = threadIdx.x;
   int ty = threadIdx.y;

   int bx = blockIdx.x;
   int by = blockIdx.y;

   int i, j, j_col, i_row;

   if (bx < stage && by < stage) {  // load upper left
     i = TILE_WIDTH * by + ty;
     j = TILE_WIDTH * bx + tx;
   }
   else if (by < stage) { // load upper right
     i = TILE_WIDTH * by + ty;
     j = TILE_WIDTH * (bx + 1) + tx;
   }
   else if (bx < stage) { // load bottom left
     i = TILE_WIDTH * (by + 1) + ty;
     j = TILE_WIDTH * bx + tx;
   }
   else { // load bottom right
     i = TILE_WIDTH * (by + 1) + ty;
     j = TILE_WIDTH * (bx + 1) + tx;
   }

   i_row = base + ty;
   j_col = base + tx;

   int index, index_row, index_col;
   index = i * size + j;
   index_row = i_row * size + j;
   index_col = i * size + j_col;


   // loads data from global memory into shared memory
   __shared__ float rowMatrix[TILE_WIDTH][TILE_WIDTH];
   __shared__ float colMatrix[TILE_WIDTH][TILE_WIDTH];

   float i_j = (i < size && j < size) ? matrix[index] : INF;
   rowMatrix[ty][tx] = (i_row < size && j < size) ? matrix[index_row] : INF;
   colMatrix[ty][tx] = (j_col < size && i < size) ? matrix[index_col] : INF;
   __syncthreads();


   if (i >= size || j >= size) return;


   // run Floyd Warshall
   float sum;
   for (int k = 0; k < TILE_WIDTH; k++) {
     sum = colMatrix[ty][k] + rowMatrix[k][tx];
     if (sum < i_j) {
       i_j = sum;
     }
    //__syncthreads();
   }

   // write back to global memory
   matrix[index] = i_j;
 }
