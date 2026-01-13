find_package(CUDAToolkit REQUIRED)

add_library(my_core_lib  hello_cuda.cu)

target_link_libraries(my_core_lib PUBLIC CUDA::cudart)
target_include_directories(my_core_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)
