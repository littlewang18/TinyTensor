#ifndef TOOLBOX
#define TOOLBOX

#include <cstddef>

template <typename T>
void matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    #pragma omp parallel for
    for(size_t i = 0; i < M; ++i) {
        for(size_t k = 0; k < K; ++k) {
            T temp = A[i * K + k];
            for(size_t j = 0; j < N; ++j) {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }
};


template <typename T>
void add(const T* A, const T* B, T* C, size_t size) {
    #pragma omp parallel for
    for(size_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
};

template <typename T>
void scale(T* A, float alpha, size_t size) {
    #pragma omp parallel for
    for(size_t i = 0; i < size; ++i) {
        A[i] *= alpha;
    }
};

template <typename T>
void func_scalar(T* A, T* B, float alpha, size_t size) {
    #pragma omp parallel for
    for(size_t i = 0; i < size; ++i) {
        B[i] = A[i] * alpha;
    }
};


template <typename T>
float mse_forward(T* A, T* B, size_t size) {
    #pragma omp parallel for
    float sum = 0.0;
    for(size_t i = 0; i < size; ++i) {
        sum += ((A[i] - B[i]) * (A[i] - B[i]));
    }
    return sum / size;
}

template <typename T>
void mse_backward(T* A, T* B, T* C, size_t size) {
    #pragma omp parallel for
    for(size_t i = 0; i < size; ++i) {
        C[i] = (2.0 / size) * (A[i] - B[i]);
    }
}


template <typename T>
void func_transpose(const T* A, T* B, size_t row, size_t col) {
    #pragma omp parallel for
    for(size_t i = 0; i < row; ++i) {
        for(size_t j = 0; j < col; ++j) {
            B[j * row + i] = A[i * col + j] ;
        }
    } 
}

// template <typename T>
// T func_sum(const T* A, size_t row, size_t col, bool axis) {
//     return 0
// }

#endif
