#include <cuda_runtime.h>
#include <stdio.h>

#define SIZE 256
#define BLOCK_SIZE 16  // 16x16 = 256 threads

__global__ void matrixMulKernel(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    float sum = 0.0f;
    if (row < width && col < width) {
        for (int i = 0; i < width; i++) {
            sum += A[row * width + i] * B[i * width + col];
        }
        C[row * width + col] = sum;
    }
}

void initMatrix(float* matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() / (float)RAND_MAX;
    }
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int matrixSize = SIZE * SIZE * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(matrixSize);
    h_B = (float*)malloc(matrixSize);
    h_C = (float*)malloc(matrixSize);

    // Initialize matrices
    initMatrix(h_A, SIZE);
    initMatrix(h_B, SIZE);

    // Allocate device memory
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);

    // Copy to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Set grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(SIZE / BLOCK_SIZE, SIZE / BLOCK_SIZE);

    // Launch kernel
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, SIZE);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Verify result (just check a few elements)
    printf("Sample results:\n");
    for (int i = 0; i < 3; i++) {
        printf("C[%d][%d] = %f\n", i, i, h_C[i * SIZE + i]);
    }

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}