cmake_minimum_required(VERSION 3.10)
project(TinyTensor VERSION 1.3  LANGUAGES CXX CUDA)

# 全局设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")


add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(app)