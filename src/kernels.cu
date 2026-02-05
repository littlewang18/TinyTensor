#include <iostream>
#include <vector>
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

__global__ void matrix_mul_native_kernel(const float* A, const float* B, float* C, int rows1, int cols1, int cols2) {
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

__global__ void matrix_mul_tiled_kernel(const float* A, const float* B, float* C, int rows1, int cols1, int cols2) {
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

__global__ void mul_num_kernel(const float* A, float* B, float alpha, int n) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i < n){
		B[i] = A[i] * alpha;
	}
}

__global__ void loss_mse(const float* A, const float* B, float* loss, int size) {
	__shared__ float temp[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
        float diff = A[i] - B[i];
        temp[threadIdx.x] = diff * diff;
    } else {
        temp[threadIdx.x] = 0.0f;
    }
	__syncthreads();
	
	// 归约
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            temp[threadIdx.x] += temp[threadIdx.x + s];
        }
        __syncthreads();
    }

	if (threadIdx.x == 0) {
        loss[blockIdx.x] = temp[0];
    }

}

__global__ void mse_backward(const float* A, const float* B, float* C, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < size) {
        C[i] = (2.0 / size) * (A[i] - B[i]);
    }
}

__global__ void transpose_kernel(const float* A, float* B, int rows, int cols) {
	__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = A[y * cols + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < rows && y < cols) {
        B[y * rows + x] = tile[threadIdx.x][threadIdx.y];
    }
}

__global__ void sum_kernel(const float* input, float* output, int rows, int cols, bool axis) {
    if (axis == 0) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        if (col < cols) {
            float sum = 0;
            for (int r = 0; r < rows; ++r) {
                sum += input[r * cols + col];
            }
            output[col] = sum;
        }
    } else {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if (row < rows) {
            float sum = 0;
            for (int c = 0; c < cols; ++c) {
                sum += input[row * cols + c];
            }
            output[row] = sum;
        }
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

void launch_mul_num(const float* A, float* B, float alpha, int n) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

	mul_num_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, alpha, n);
}

float launch_mse_forward(const float* A, const float* B, int size) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

	float* loss;
    cudaMalloc(&loss, blocksPerGrid * sizeof(float));

	loss_mse<<<blocksPerGrid, threadsPerBlock>>>(A, B, loss, size);

    std::vector<float> losses(blocksPerGrid);
    cudaMemcpy(losses.data(), loss, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost);

    float total_loss = 0.0f;
    for (float val : losses) {
        total_loss += val;
    }

    cudaFree(loss);
    return total_loss / size;
}

void launch_mse_backward(const float* A, const float* B, float* C, int size) {
	int threadsPerBlock = 256;
	int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
	
	mse_backward<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, size);
}

void launch_transpose(const float* A, float* B, int rows, int cols) {
	dim3 threadsPerBlock(16, 16);
	dim3 blocksPerGrid((cols + 15) / 16, (rows + 15) / 16);

	transpose_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, rows, cols);
}

void launch_sum(const float* A, float* B, int rows, int cols, bool axis) {
	int  threadsPerBlock = 256;
	int blocksPerGrid = 0;
	if (axis == 0) {
        blocksPerGrid = (cols + threadsPerBlock - 1) / threadsPerBlock;
    } else {
        blocksPerGrid = (rows + threadsPerBlock - 1) / threadsPerBlock;
    }

	sum_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, rows, cols, axis);
}