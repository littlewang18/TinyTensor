#include "cuda_bridge.h"
#include <cuda_runtime.h>
#include <iostream>

void* gpu_malloc(size_t size) {
    void* ptr;
    if (cudaMalloc(&ptr, size) != cudaSuccess) return nullptr;
    return ptr;
}

void gpu_free(void* ptr) {
    if (ptr) cudaFree(ptr);
}

void gpu_memcpy_h2d(void* dest, const void* src, size_t size) {
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void gpu_memcpy_d2h(void* dest, const void* src, size_t size) {
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}