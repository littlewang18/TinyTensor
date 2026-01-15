#ifndef TOOLBOX
#define TOOLBOX

template <typename T>
void matmul(const TinyTensor<T>& A, const TinyTensor<T>& B, TinyTensor<T>& C, int N) {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            T temp = A(i, k);
            for (int j = 0; j < N; ++j) {
                C(i, j) += temp * B(k, j);
            }
        }
    }
};


#endif
