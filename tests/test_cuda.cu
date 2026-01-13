#include <iostream>
#include <TinyTensor.h>
#include <cuda_runtime.h>
#include <kernels.h>

// CUDA测试
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
	launch_vector_add(d_a, d_b, d_c, n);

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

// TinyTensor CUDA测试
void test_TinyTensor_cuda(const int& N) {
    TinyTensor<float> A(1, N), B(1, N), C(1, N);

    A.fill(1.0f);
    B.fill(2.0f);

    A.allocate_device(); A.copy_to_device();
    B.allocate_device(); B.copy_to_device();
    C.allocate_device();

    // 调用 kernel
    launch_vector_add(A.get_cuda_ptr(), B.get_cuda_ptr(), C.get_cuda_ptr(), N);

    C.copy_to_host();

    std::cout << "Result C[0]: " << C[0] << std::endl;
}


int main() {
    // std::cout << "\n---Testing CUDA ---" << std::endl;
    // launch_hello_kernel();
    // test_vector_add();
    
    std::cout << "\n---Testing TinyTensor CUDA ---" << std::endl;
    test_TinyTensor_cuda(1000000);
    
    return 0;
}