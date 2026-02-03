#ifndef TOOLBOX
#define TOOLBOX

template <typename T>
void matmul(const TinyTensor<T>& A, const TinyTensor<T>& B, TinyTensor<T>& C, size_t M, size_t K, size_t N) {
    #pragma omp parallel for
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K ++k) {
            T temp = A(i, k);
            for (int j = 0; j < N; ++j) {
                C(i, j) += temp * B(k, j);
            }
        }
    }
};


template <typename T>
void add(const TinyTensor<T>& A, const TinyTensor<T>& B, TinyTensor<T>& C, size_t row, size_t col) {
    #pragma omp parallel for
    for (int i = 0; i < row; ++i) {
        for (int j = 0; j < col; ++j) {
            C(i, j) = A(i, j) + B(i, j);
        }
    }
};

template <typename T>
void scale(const TinyTensor<T>& A, float alpha, size_t size) {
    #pragma omp parallel for
    for (int i = 0; i < size; ++i) {
        A[i] *= alpha;
    }
};


#endif
