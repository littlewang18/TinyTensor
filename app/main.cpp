#include <iostream>
#include "Linear.h"
#include "SGD.h"

int main() {
    TinyTensor<float> x(1, 1); x[0] = 1.0f;
    TinyTensor<float> y_true(1, 1); y_true[0] = 9.0f;
    
    // 2. 初始化模型
    Linear<float> model(1, 1); 
    
    model.to("gpu");
    x.to("gpu");
    y_true.to("gpu");

    std::vector<TinyTensor<float>*> params = model.get_parameters(); 
    SGD<float> optimizer(0.01f); 

    std::cout << "开始训练..." << std::endl;

    x.set_grad(1);

    for (int epoch = 0; epoch < 200; ++epoch) {
        // 前向传播 
        auto y_pred = model.forward(x);
        
        // 计算 MSE 梯度
        y_pred.set_grad(true);
        y_pred.get_grad() = (y_pred + (y_true * -1.0f)) * 0.5f;
        
        // 反向传播
        model.backward(x, y_pred);
        
        // 参数更新
        optimizer.step(params);
        
        // 结果
        if (epoch % 20 == 0) {
            y_pred.to("cpu");
            float loss = 0;
            for(int i=0; i<4; ++i) {
                float diff = y_pred[i] - y_true[i];
                loss += diff * diff;
            }
            std::cout << "Epoch " << epoch << " MSE Loss: " << loss / 4.0f << std::endl;
        }
    }
    
    std::cout << "训练完成！" << std::endl;
    return 0;
}
