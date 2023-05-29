#include <cuda_runtime.h>
#include <iostream>
#define N 1000

__global__ void matrixMul(int *a, int *b, int *c)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        int sum = 0;
        for (int k = 0; k < N; k++)
        {
            sum += a[row * N + k] * b[k * N + col];
        }
        c[row * N + col] = sum;
    }
}

int main()
{
    int *a, *b, *c;
    // host matrices
    int *dev_a, *dev_b, *dev_c;
    // device matrices
    int size = N * N * sizeof(int);

    // Allocate memory on host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Allocate memory on device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Initialize host matrices
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            a[i * N + j] = i;
            b[i * N + j] = j;
        }
    }

    // Copy host matrices to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 1024 threads per block
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (N + blockDim.y - 1) / blockDim.y);
    matrixMul<<<gridDim, blockDim>>>(dev_a, dev_b, dev_c);

    // Copy result from device to host
    cudaMemcpy(c, dev_c, size, cudaMemcpyDeviceToHost);

    // Print the first 10 values of the result
    for (int i = 0; i < 10; i++)
    {
        std::cout << c[i] << " ";
    }

    // Free memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(a);
    free(b);
    free(c);

    return 0;
}