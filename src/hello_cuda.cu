#include <iostream>
#include <cuda_runtime.h>


__global__ void hello_from_gpu() {
	int tid = threadIdx.x;
	printf("Hello World form GPU thread %d! \n", tid);
}

__global__ void vector_add_kernel(const float* a, const float* b, float* c, int n) {
	int i = threadIdx.x;
	if (i < n) {
		c[i] = a[i] + b[i];
	}
}


void launch_hello_kernel() {
	hello_from_gpu<<<1, 10>>>();

	cudaDeviceSynchronize();
}


void launch_vector_add(const float* a, const float* b, float* c, int n) {
	vector_add_kernel<<<1, 10>>>(a, b, c, n);

	cudaDeviceSynchronize();
}


void test_vector_add() {
	int n = 5;
	float h_a[] = {1.0, 2.0, 3.0, 4.0, 5.0};
	float h_b[] = {10.0, 20.0, 30.0, 40.0, 50.0};
	float h_c[5];

	float *d_a, *d_b, *d_c;

	// 申请内存
	cudaMalloc(&d_a, n * sizeof(float));
	cudaMalloc(&d_b, n * sizeof(float));
	cudaMalloc(&d_c, n * sizeof(float));

	// 从Host 拷贝到 Device
	cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice);
	
	//启动核函数
	vector_add_kernel<<<1, n>>>(d_a, d_b, d_c, n);

	// 从Device拷贝回Host
	cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);

	// 打印
	std::cout << "GPU Result: ";
	for(int i=0; i<n; i++) std::cout << h_c[i] <<" ";
	std::cout << std::endl;

	//释放显存
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
}

