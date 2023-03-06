
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>



// CUDA kernel for Vector Addition
__global__ void vectorAdd(int* vec_a, int* vec_b, int* vec_c, int vector_size) {
    // Calculate global thread ID (tid)
    int tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Vector boundary guard
    if (tid < vector_size) {
        // Each thread adds a single element
        vec_c[tid] = vec_a[tid] + vec_b[tid];
    }
}

void matrix_init(int* vector, int vector_size) {
    for (int i = 0; i < vector_size; i++) {
        vector[i] = rand() % 100;
    }
}

// Check VectorAdd result
void error_check(int* a, int* b, int* c, int vector_size) {
    for (int i = 0; i < vector_size; i++) {
        assert(c[i] == a[i] + b[i]);
    }
}

int main() {
    // Vector of size 2^16 (~65K elements)
    int vector_size = 1234;
    // Host vector pointers
    int* h_a, * h_b, * h_c;
    // Device vector pointers
    int* d_a, * d_b, * d_c;

    // Allocation size for all vectors
    size_t total_bytes = sizeof(int) * vector_size;

    // Allocate host memory

    h_a = (int*)malloc(total_bytes);
    h_b = (int*)malloc(total_bytes);
    h_c = (int*)malloc(total_bytes);

    // Allocate device memory
    cudaMalloc(&d_a, total_bytes);
    cudaMalloc(&d_b, total_bytes);
    cudaMalloc(&d_c, total_bytes);

    // Initialize vectors a and b with random value between 0 and 99
    matrix_init(h_a, vector_size);
    matrix_init(h_b, vector_size);

    // Copy data from CPU to GPU
    cudaMemcpy(d_a, h_a, total_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, total_bytes, cudaMemcpyHostToDevice);

    // Block Size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCKS = (int)ceil(vector_size / NUM_THREADS);

    // Launch Kernel on default stream without shared memory
    vectorAdd << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, vector_size);

    // Copy the sum of vector from GPU to CPU
    cudaMemcpy(h_c, d_c, total_bytes, cudaMemcpyDeviceToHost);

    // Check result for errors
    error_check(h_a, h_b, h_c, vector_size);

    printf("Completed Successfully\n");
    return 0;
}