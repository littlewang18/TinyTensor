#ifndef KERNELS
#define KERNELS


void launch_hello_kernel();

void launch_vector_add(const float* a, const float* b, float* c, const int& n);

void launch_matrix_add(const float* A, const float* B, float* C, int rows, int cols);

void launch_matrix_mul_native(const float* A, const float* B, float* C, int rows1, int cols1, int cols2);

void launch_matrix_mul_tiled(const float* A, const float* B, float* C, int rows1, int cols1, int cols2);

void launch_relu(float* data_device, int n);

void launch_scale(float* data_device, float alpha, int n);

#endif