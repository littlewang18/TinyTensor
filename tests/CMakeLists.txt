add_executable(test_cpu_ops test_cpu.cpp)
add_executable(test_cuda_ops test_cuda.cu)

target_link_libraries(test_cpu_ops PRIVATE my_core_lib)
target_link_libraries(test_cuda_ops PRIVATE my_core_lib)