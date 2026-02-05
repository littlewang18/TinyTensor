#ifndef SGDH
#define SGDH

#include <vector>
#include "Optimizer.h"
#include "TinyTensor.h"


template <typename T>
class SGD : public Optimizer<T> {
    public:
        SGD(T lr) : Optimizer<T>(lr) {}

        void step(std::vector<TinyTensor<T>*>& params) override {
            for (auto* p : params) {
                if (p->get_status() == DeviceStatus::DeviceOnly || p->get_status() == DeviceStatus::Synced) {                    
                    *p = *p + (p->get_grad() * (-this->learning_rate));
                } else {
                    T* data = p->get_ptr();
                    T* grad = p->get_grad().get_ptr();
                    for (size_t i = 0; i < p->get_size(); ++i) {
                        data[i] -= this->learning_rate * grad[i];
                    }
                }
            }
        }
};

#endif