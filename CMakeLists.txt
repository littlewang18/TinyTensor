cmake_minimum_required(VERSION 3.10)
project(TinyTensor VERSION 1.0)

# 全局设置
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(src)
add_subdirectory(app)
