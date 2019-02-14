#pragma once
#include "FloydWarshall.cuh"
#include <iostream>
#include <limits>

#define BLOCK_SIZE 32

__constant__ auto INF = std::numeric_limits<float>::infinity();   // qui andrebbe sistemato in modo che al posto di float accetti T

////////////////////////////////////////////////////////////////////////////////
//! Naive floyd_warshall kernel implementation
//! @param d_N  input data in global memory
//! @param n  number of verticies of the input matrix N
//! @param k  index of the intermediate vertex
////////////////////////////////////////////////////////////////////////////////
__global__ void naive_floyd_warshall_kernel(int *matrix, int size, int k) {
    // compute indexes
    const unsigned int i = blockIdx.y;
    const unsigned int j = blockIdx.x;

    // read in dependent values
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
  if (i >= n || j >= n || k >= n || i == j) return;

  const float i_k_value = N[i * n + k];
  const float k_j_value = N[k * n + j];
  const float i_j_value = N[i * n + j];

  if (i_k_value != INF && k_j_value != INF) {
      float sum = i_k_value + k_j_value;
      if (sum < i_j_value) {
          N[i * n + j] = sum;
      }
  }
}
