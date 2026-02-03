#ifndef TOOLBOX
#define TOOLBOX

#include <cstddef>

template <typename T>
void matmul(const T* A, const T* B, T* C, size_t M, size_t K, size_t N) {
    #pragma omp parallel for
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            T temp = A[i * K + k];
            for (size_t j = 0; j < N; ++j) {
                C[i * N + j] += temp * B[k * N + j];
            }
        }
    }
};


template <typename T>
void add(const T* A, const T* B, T* C, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        C[i] = A[i] + B[i];
    }
};

template <typename T>
void scale(T* A, float alpha, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        A[i] *= alpha;
    }
};


#endif
