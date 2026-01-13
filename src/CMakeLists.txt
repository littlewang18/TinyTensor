add_library(my_core_lib  
    kernels.cu 
    cuda_bridge.cu
)

target_include_directories(my_core_lib PUBLIC ${PROJECT_SOURCE_DIR}/include)