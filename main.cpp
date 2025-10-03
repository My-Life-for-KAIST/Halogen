#include "Halogen/Core/Tensor.h"
#include "Halogen/Core/TensorOperation.h"
#include <iostream>

using namespace Halogen::TensorOperation;

int main() {
    Halogen::Tensor<int> T({1,2,3,4,5,6,7,8}, {4,2});
    auto ok =
            bind(T)
            >> reshape({8})
            >> add(3);

    std::cout << T[0] << std::endl;
    std::cout << (ok ? "ok\n" : "fail\n");
}