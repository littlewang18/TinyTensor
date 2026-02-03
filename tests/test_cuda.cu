#include <iostream>
#include <TinyTensor.h>
#include <cuda_runtime.h>
#include <kernels.h>
#include <chrono>
#include <toolbox.h>

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
    TinyTensor<float> A(N, N), B(N, N);

    A.fill(1.0f);
    B.fill(2.0f);

    A.allocate_device(); A.copy_to_device();
    B.allocate_device(); B.copy_to_device();

    // 调用 kernel
    // launch_vector_add(A.get_cuda_ptr(), B.get_cuda_ptr(), C.get_cuda_ptr(), N);
    // 调用自身
    TinyTensor<float> C = A * B;
    C.copy_to_host();

    std::cout << "Result C[0]: " << C[0] << std::endl;
}

// TinyTensor 加法测试
void test_matrix_add_2d() {
    int rows = 1000;
    int cols = 1000;
    TinyTensor<float> A(rows, cols), B(rows, cols), C(rows, cols);

    A.fill(1.5f);
    B.fill(2.5f);

    A.allocate_device(); A.copy_to_device();
    B.allocate_device(); B.copy_to_device();
    C.allocate_device();

    // 启动 2D 算子
    launch_matrix_add(A.get_cuda_ptr(), B.get_cuda_ptr(), C.get_cuda_ptr(), rows, cols);

    C.copy_to_host();

    // 验证结果
    std::cout << "Result: " << C(500, 500) << std::endl;
}

// TinyTensor CPU测试
void test_matrix_mul_CPU() {
    int rows = 1024;
    int cols = 1024;
    TinyTensor<float> A(rows, cols), B(rows, cols), C(rows, cols);

    A.fill(2.0f);
    B.fill(3.0f);

    // CPU OMP版本
    auto start = std::chrono::high_resolution_clock::now();
    matmul(A, B, C, rows);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout << "CPU版本："<< diff.count() << " seconds" << std::endl;

    // 验证结果
    std::cout << "Result: " << C(500, 500) << std::endl;
}

// TinyTensor GPU测试
void test_matrix_mul_GPU_native() {
    int rows = 1024;
    int cols = 1024;
    TinyTensor<float> A(rows, cols), B(rows, cols), C(rows, cols);

    A.fill(2.0f);
    B.fill(3.0f);

    A.allocate_device(); A.copy_to_device();
    B.allocate_device(); B.copy_to_device();
    C.allocate_device();

    // native乘法
    auto start = std::chrono::high_resolution_clock::now();
    launch_matrix_mul_native(A.get_cuda_ptr(), B.get_cuda_ptr(), C.get_cuda_ptr(), rows, cols, cols);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout<< " GPU普通版：" << diff.count() << " seconds" << std::endl;
    
    C.copy_to_host();

    // 验证结果
    std::cout << "Result: " << C(500, 500) << std::endl;
}

// TinyTensor GPU 分块测试
void test_matrix_mul_GPU_tiled() {
    int rows = 1024;
    int cols = 1024;
    TinyTensor<float> A(rows, cols), B(rows, cols), C(rows, cols);

    A.fill(2.0f);
    B.fill(3.0f);

    A.allocate_device(); A.copy_to_device();
    B.allocate_device(); B.copy_to_device();
    C.allocate_device();

    // 分块乘法
    auto start = std::chrono::high_resolution_clock::now();
    launch_matrix_mul_tiled(A.get_cuda_ptr(), B.get_cuda_ptr(), C.get_cuda_ptr(), rows, cols, cols);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    std::cout<< " GPU分块版：" << diff.count() << " seconds" << std::endl;

    C.copy_to_host();

    // 验证结果
    std::cout << "Result: " << C(500, 500) << std::endl;
}

// Relu测试
void test_elementwise() {
    TinyTensor<float> A(10, 10);
    A.fill(-1.0f); // 全是负数
    A.to("cuda");

    A *= 0.5f;  // 整个矩阵每个元素都减半
    A.relu();   // 执行 GPU ReLU

    A.to("cpu");
    std::cout << "ReLU 结果: " << A(5, 5) << std::endl;
}


int main() {
    // std::cout << "\n---Testing CUDA ---" << std::endl;
    // launch_hello_kernel();
    // test_vector_add();
    
    std::cout << "\n---Testing TinyTensor CUDA ---" << std::endl;
    // test_TinyTensor_cuda(1024);
	// test_matrix_add_2d();  
    // test_matrix_mul_CPU();
    // test_matrix_mul_GPU_native();
    // test_matrix_mul_GPU_tiled();

    test_elementwise();

    return 0;
}