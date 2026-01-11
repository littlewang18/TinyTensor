#include <TinyTensor.h>
#include <iostream>
#include <utility>

int main() {
    // 1. 测试模板和智能指针管理
    TinyTensor<float> t1(1024); // 申请 1024 个 float 的空间
    t1.fill(1.5f);

    // 2. 测试移动语义
    std::cout << "t1 移动前地址: " << t1.data_() << std::endl;
    
    TinyTensor<float> t2 = std::move(t1); // 触发移动构造
    
    std::cout << "t2 接收后的地址: " << t2.data_() << std::endl;
    
    // 验证 t1 是否变为空
    if (t1.data_() == nullptr) {
        std::cout << "移动成功：t1 已释放所有权。" << std::endl;
    }

    return 0;
}