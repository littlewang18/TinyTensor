#include <TinyTensor.h>
#include <iostream>
#include <chrono>

extern void launch_hello_kernel();
extern void launch_vector_add(const float* a, const float* b, float* c, int n);
extern void test_vector_add();

//矩阵乘法函数：C = A * B
#pragma omp parallel for
template <typename T>
void matmul(TinyTensor<T>& A, TinyTensor<T>& B, TinyTensor<T>& C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            T temp = A(i, k);
            for (int j = 0; j < N; ++j) {
                C(i, j) += temp * B(k, j);
            }
        }
    }
}

void test_TinyTensor() {
    std::cout << "--- Correctness Test (3x3) ---" << std::endl;
    int N_small = 3;
    TinyTensor<float> A_s(N_small, N_small), B_s(N_small, N_small), C_s(N_small, N_small);
    
    A_s.fill(1.0f); // 全 1 矩阵
    B_s.fill(2.0f); // 全 2 矩阵
    
    matmul(A_s, B_s, C_s, N_small);
    C_s.print(); // 结果应该全是 6.0 (1*2 + 1*2 + 1*2)

    // --- 2. 性能压力测试 (1024x1024 矩阵) ---
    std::cout << "\n--- Performance Test (1024x1024) ---" << std::endl;
    int N = 1024;
    TinyTensor<float> A(N, N), B(N, N), C(N, N);
    A.fill(1.1f);
    B.fill(2.2f);

    auto start = std::chrono::high_resolution_clock::now();
    
    matmul(A, B, C, N);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    std::cout << "Matrix Multiplication (Size " << N << "x" << N << ") took: " 
              << diff.count() << " seconds" << std::endl;
}

void test_TinyTensor_cuda() {
    int N = 100;
    TinyTensor<float> A(1, N), B(1, N), C(1, N);

    A.fill(1.0f);
    B.fill(2.0f);

    A.allocate_device(); A.copy_to_device();
    B.allocate_device(); B.copy_to_device();
    C.allocate_device();

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // 调用 kernel
    launch_vector_add(A.get_cuda_ptr(), B.get_cuda_ptr(), C.get_cuda_ptr(), N);

    C.copy_to_host();

    // 验证：C[0] 应该是 3.0
    std::cout << "Result C[0]: " << C[0] << std::endl;
}

int main() {
    
    // std::cout << "--- Correctness TinyTensor ---" << std::endl;
    // test_TinyTensor();


    // std::cout << "\n---Testing CUDA ---" << std::endl;
    // launch_hello_kernel();
    // test_vector_add();
    
    std::cout << "\n---Testing TinyTensor CUDA ---" << std::endl;
    test_TinyTensor_cuda();
    
    return 0;
}