#include <iostream>
#include "Linear.h"

int main() {
    Linear<float> fc1(784, 128);
    Linear<float> fc2(128, 10);

    auto x = TinyTensor<float>(4, 784); 
    x.fill(0.5f);
    
    x.to("gpu");
    fc1.to("gpu");
    fc2.to("gpu");
    
    auto h1 = fc1.forward(x);
    auto out = fc2.forward(h1);

    out.to("cpu");
    out.print();

    return 0;
}
