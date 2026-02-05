#ifndef CUDA_BRIDGE_H
#define CUDA_BRIDGE_H

#include <cstddef>

void*gpu_malloc(size_t size);
void gpu_free(void* ptr);
void gpu_memcpy_h2d(void* dest, const void* src, size_t size);
void gpu_memcpy_d2h(void* dest, const void* src, size_t size);
void gpu_memset_zero(void* ptr, size_t size);

#endif