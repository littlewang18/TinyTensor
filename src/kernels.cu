#include <iostream>
#include <cuda_runtime.h>
#include <kernels.h>

#define TILE_SIZE 16

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

__global__ void matrix_add_kernel(const float* A, const float* B, float* C, int rows, int cols) {
	int r = blockIdx.y * blockDim.y + threadIdx.y;
	int c = blockIdx.x * blockDim.x + threadIdx.x;

	// 边界检测
	if(r < rows && c < cols) {
		int idx = r * cols + c;
		C[idx] = A[idx] + B[idx];
	}
}

__global__ void matrix_mul_native_kernel(const float* A, const float* B, float* C,  int rows1, int cols1, int cols2) {
	int row = blockIdx.y * blockDim.y + threadIdx.y; // 行
	int col = blockIdx.x * blockDim.x + threadIdx.x; // 列

	if(row < rows1 && col < cols2) {
		float sum {0.0f};
		for(int i = 0; i < cols1; ++i) {
			sum += A[row * cols1 + i] * B[i * cols2 + col];
		}
		C[row * cols2 + col] = sum;
	}

}

__global__ void matrix_mul_tiled_kernel(const float* A, const float* B, float* C,  int rows1, int cols1, int cols2) {
	__shared__ float tile_A[TILE_SIZE][TILE_SIZE];
	__shared__ float tile_B[TILE_SIZE][TILE_SIZE];

	int row = blockIdx.y * TILE_SIZE + threadIdx.y;
	int col = blockIdx.x * TILE_SIZE + threadIdx.x;

	float sum = 0.0f;

	for (int m = 0; m < (cols1 + TILE_SIZE - 1) / TILE_SIZE; ++m) {
		if(row < rows1 && m * TILE_SIZE + threadIdx.x < cols1)
			tile_A[threadIdx.y][threadIdx.x] = A[row * cols1 + m * TILE_SIZE +threadIdx.x];
		else
			tile_A[threadIdx.y][threadIdx.x] = 0.0f;
		
		if(col < cols2 && m * TILE_SIZE + threadIdx.y < cols1)
			tile_B[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * cols2 + col];
		else
			tile_B[threadIdx.y][threadIdx.x] = 0.0f;

		__syncthreads();

		for(int k = 0; k < TILE_SIZE; ++k) {
			sum += tile_A[threadIdx.y][k] * tile_B[k][threadIdx.x];
		}
		__syncthreads();
	}

	if(row < rows1 && col < cols2) {
		C[row * cols2 + col] = sum;
	}
}

__global__ void relu_kernel(float* data, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		data[i] = data[i] > 0 ? data[i] : 0;
	}
}

__global__ void scale_kernel(float* data, float alpha, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		data[i] = data[i] * alpha;
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


void launch_matrix_add(const float* A, const float* B, float* C, int rows, int cols) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
					  (rows + threadsPerBlock.y - 1) / threadsPerBlock.y);
	
	matrix_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rows, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA Error: %s\n", cudaGetErrorString(err));
}


void launch_matrix_mul_native(const float* A, const float* B, float* C, int rows1, int cols1, int cols2) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols1 + 15) / 16, (rows1 + 15) / 16);

	matrix_mul_native_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rows1, cols1, cols2);
	cudaDeviceSynchronize();
}


void launch_matrix_mul_tiled(const float* A, const float* B, float* C, int rows1, int cols1, int cols2) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols2 + 15) / 16, (rows1 + 15) / 16);

	matrix_mul_tiled_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, rows1, cols1, cols2);
}


void launch_relu(float* data_device, int n) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(data_device, n);
}


void launch_scale(float* data_device, float alpha, int n) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	scale_kernel<<<blocksPerGrid, threadsPerBlock>>>(data_device, alpha, n);
}