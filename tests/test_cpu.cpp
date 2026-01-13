#include <TinyTensor.h>
#include <toolbox.h>
#include <iostream>
#include <chrono>


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


int main() {
    std::cout << "--- Correctness TinyTensor ---" << std::endl;
    test_TinyTensor();
    
    return 0;
}