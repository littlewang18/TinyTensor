#include <iostream>


int main() {
    const int constant = 26;
    const int* const_p = &constant;
    int* modifier = const_cast<int*>(const_p);
    *modifier = 3;
    std::cout<< "constant:  "<<constant<<"  adderss:  "<< &constant <<std::endl;
    std::cout<<"*modifier:  "<<*modifier<<"  adderss:  " << modifier<<std::endl;

    return 0;
}
