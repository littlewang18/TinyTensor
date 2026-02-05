#ifndef OPTIMIZER
#define OPTIMIZER

#include <vector>
#include "TinyTensor.h"


template <typename T>
class Optimizer {
    protected:
        T learning_rate;
    public:
        Optimizer(T lr) : learning_rate(lr) {}
        virtual void step(std::vector<TinyTensor<T>*>& params) = 0;
        virtual ~Optimizer() = default;
};

#endif