
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <stdexcept>



// CUDA kernel for Vector Multiplication
__global__ void multiply(int* vec_a, int* vec_b, int* vec_c, int vector_size) {
    // Calculate global thread ID (tid)
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
    // Vector boundary guard
    if (idx < vector_size) {
        // Each thread adds a single element
        vec_c[idx] = vec_a[idx] * vec_b[idx];
    }
}

// CUDA kernel for Vector Division
__global__ void divide(int* vec_a, int* vec_b, int* vec_c, int vector_size) {
    // Calculate global thread ID (tid)
    int idx = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Vector boundary guard
    if (idx < vector_size) {
        // Each thread adds a single element
        vec_c[idx] = vec_a[idx] / vec_b[idx];
    }
}

void vector_init(int* vector, int vector_size) {
    for (int i = 0; i < vector_size; i++) {
        vector[i] = (rand() % 99) + 1;
    }
}

// Check VectorAdd result
void error_check(int* a, int* b, int* c, int vector_size, char operation_type) {
    if (operation_type == '*') {
        for (int i = 0; i < vector_size; i++) {
            if (c[i] != a[i] * b[i]) {
                printf("Error in computation: %d != %d * %d (%d)", c[i], a[i], b[i], a[i] * b[i]);
                throw std::runtime_error("Error in Computation");
            }
        }
    }
    else {
        for (int i = 0; i < vector_size; i++) {
            if (c[i] != a[i] / b[i]) {
                printf("Error in computation: %d != %d / %d (%d)", c[i], a[i], b[i], a[i] / b[i]);
                throw std::runtime_error("Error in Computation");
            }
        }
    }
}

void compute_for_specific_size(int* d_a, int* d_b, int* d_c, int vector_size, char operation_type) {
    // Block Size
    int NUM_THREADS = 256;

    // Grid size
    int NUM_BLOCKS = (int)ceil(vector_size / NUM_THREADS);

    // Launch Kernel on default stream without shared memory
    if (operation_type == '*') {
        multiply << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, vector_size);
    }
    else {
        divide << <NUM_BLOCKS, NUM_THREADS >> > (d_a, d_b, d_c, vector_size);
    }
    cudaDeviceSynchronize();
}

int main() {
    cudaSetDevice(0);
    // Vector of size 2^16 (~65K elements)
    int max_vector_size = 2;
    // Host vector pointers
    int* h_a, * h_b, * h_c;
    // Device vector pointers
    int* d_a, * d_b, * d_c;
    for (int current_vector_size = 1; current_vector_size < max_vector_size; current_vector_size += 1) {

        // Allocation size for a single vector if size vector_size
        size_t vector_bytes = sizeof(int) * current_vector_size;
        // Allocate host memory
        h_a = (int*)malloc(vector_bytes);
        h_b = (int*)malloc(vector_bytes);
        h_c = (int*)malloc(vector_bytes);

        // Allocate device memory
        cudaMalloc(&d_a, vector_bytes);
        cudaMalloc(&d_b, vector_bytes);
        cudaMalloc(&d_c, vector_bytes);

        // Initialize vectors a and b with random value between 0 and 99
        vector_init(h_a, current_vector_size);
        vector_init(h_b, current_vector_size);

        // Copy data from CPU to GPU
        cudaMemcpy(d_a, h_a, vector_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h_b, vector_bytes, cudaMemcpyHostToDevice);

        compute_for_specific_size(d_a, d_b, d_c, current_vector_size, '*');

        // Copy the sum of vector from GPU to CPU
        cudaMemcpy(h_c, d_c, vector_bytes, cudaMemcpyDeviceToHost);
        printf("h_a: %d\nh_b: %d\nh_c: %d\n", h_a[0], h_b[0], h_c[0]);
        // Check result for errors
        error_check(h_a, h_b, h_c, current_vector_size, '*');
        free(h_a);
        free(h_b);
        free(h_c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }

    printf("Completed Successfully\n");

    return 0;
}