#include <cuda_runtime.h>
#include <iostream>
#define N 1000000

__global__ void add(int *a, int *b, int *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    int *a, *b, *c;

    // host vectors
    int *dev_a, *dev_b, *dev_c;
    // device vectors
    int size = N * sizeof(int);

    // Allocate memory on host
    a = (int *)malloc(size);
    b = (int *)malloc(size);
    c = (int *)malloc(size);

    // Allocate memory on device
    cudaMalloc((void **)&dev_a, size);
    cudaMalloc((void **)&dev_b, size);
    cudaMalloc((void **)&dev_c, size);

    // Initialize host vectors
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i * 2;
    }

    // Copy host vectors to device
    cudaMemcpy(dev_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block and 1024 threads per block
    add<<<1, 1024>>>(dev_a, dev_b, dev_c);

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