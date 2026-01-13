#include <iostream>
#include <cuda_runtime.h>
#include <kernels.h>

__global__ void hello_from_gpu() {
	int tid = threadIdx.x;
	printf("Hello World form GPU thread %d! \n", tid);
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}


void launch_hello_kernel() {
	hello_from_gpu<<<1, 10>>>();

	cudaDeviceSynchronize();
}


void launch_vector_add(const float* a, const float* b, float* c,  const int& n) {
	int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);

	cudaDeviceSynchronize();
}