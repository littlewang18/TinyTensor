#ifndef LINEAR
#define LINEAR

#include <cmath>
#include <random>
#include <string>
#include "TinyTensor.h"


template <typename T>
class Linear {
    private:
        TinyTensor <T> weight;
        TinyTensor <T> bias;

    public:
        Linear(int num_in, int num_out): 
            weight(num_in, num_out), bias(4, num_out) {
            // Xavier 初始化
            double var_weight = std::sqrt(2.0 / (num_in + num_out));

            std::default_random_engine generator(std::random_device{}());
            std::normal_distribution<T> distribution(0, (T)var_weight);
            
            size_t size_weight = weight.get_size();
            for(int i = 0; i < size_weight; ++i) { 
                weight[i] = distribution(generator);
            }
            bias.fill(0.0f);
        }
        
        // 前向传播
        TinyTensor<T> forward(const TinyTensor<T>& input) {
            auto output = (input * weight) + bias;
            output.relu();
            return output;
        }

        // 均方损失
        


        
        // CUDA适配
        Linear& to(const std::string& device) {
            if(device =="cuda" || device =="gpu") {
				weight.sync_to_device();
                bias.sync_to_device();
			} else if(device =="cpu") {
				weight.sync_to_host();
                bias.sync_to_host();
			} else {
				std::cerr << "Unknow device:" << device << '\n';
			}
			return *this;
		}
};

#endif