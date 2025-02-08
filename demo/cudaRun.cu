#include <stdio.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <chrono>
#include <iomanip>
#include <iostream>

using namespace std::chrono;
#define N 1000000056  // Define the size of the array

struct Timer {
    high_resolution_clock::time_point start_time;
    Timer() { start_time = high_resolution_clock::now(); }
    void elapsed() {
        auto end_time = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end_time - start_time);
        long long ms = duration.count();
        printf("%lld minutes, %lld seconds, %lld milliseconds\n", 
            ms / 60000, (ms / 1000) % 60, ms % 1000);
    }
    void reset() { start_time = high_resolution_clock::now(); }
};

#define THREADS_PER_BLOCK 256
#define BLOCKS ((N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK)

__global__ void matrixMultiply(float* input, float scalar, float* output) {
    unsigned long long idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        // Each thread processes one element
        output[idx] = input[idx] * scalar;
        // Add synchronization to ensure accurate timing
        __syncthreads();
        
        // Only thread 0 of block 0 prints timing (to avoid multiple prints)
        if (idx == 0) {
            clock_t start = clock64();
            output[idx] = input[idx] * scalar;
            clock_t end = clock64();
            float milliseconds = (end - start) * 1000.0f / clock64();
            printf("Kernel iteration time: %.3f milliseconds\n", milliseconds);
        }
    }
}

int main() {
    float *d_input, *d_output;
    curandGenerator_t gen;
    
    // Allocate device memory
    cudaError_t malloc_error;
    malloc_error = cudaMalloc(&d_input, N * sizeof(float));
    if (malloc_error != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(malloc_error));
        return -1;
    }
    malloc_error = cudaMalloc(&d_output, N * sizeof(float));
    if (malloc_error != cudaSuccess) {
        printf("CUDA malloc error: %s\n", cudaGetErrorString(malloc_error));
        cudaFree(d_input);
        return -1;
    }
    
    // Initialize CURAND generator
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
    curandGenerateUniform(gen, d_input, N);
    
    // Set scalar multiplier
    float scalar = 2.0f;
    
    // Create timer and launch kernel
    Timer timer;
    matrixMultiply<<<BLOCKS, THREADS_PER_BLOCK>>>(d_input, scalar, d_output);
    
    // Synchronize device
    cudaDeviceSynchronize();
    
    // Check for errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    // Clean up
    curandDestroyGenerator(gen);
    cudaFree(d_input);
    cudaFree(d_output);
    
    printf("Matrix multiplication completed successfully\n");
    return 0;
}
