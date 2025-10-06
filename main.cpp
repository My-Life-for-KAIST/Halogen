#include "Halogen/Core/Tensor.h"
#include "Halogen/Core/TensorOperation.h"
#include <iostream>

using namespace Halogen::TensorOperation;

int main() {
    auto T2 = Halogen::Tensor<int>::identity(2,2).value();
    auto T3 = Halogen::Tensor<int>({1,2,3,4}, {2,2});

    auto ok2 =
        bind(T3)
        >> matmul(T3);

    auto T4 = Halogen::Tensor<int>::arange(5);
    auto T5 = Halogen::Tensor<int>::arange(1,5);
    auto T6 = Halogen::Tensor<int>::arange(1,7,2);
    auto T7 = Halogen::Tensor<int>::arange(1,7, -1);

    std::cout << "$ T5 " << (T5 ? "ok" : "err") << std::endl;
    std::cout << "$ T6 " << (T6 ? "ok" : "err") << std::endl;
    std::cout << "$ T7 " << (T7 ? "ok" : "err") << std::endl;

    std::cout << "T4: ";
    for (int i = 0; i < T4.value().size(); i++) {
        std::cout << T4.value()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "T5: ";
    for (int i = 0; i < T5.value().size(); i++) {
        std::cout << T5.value()[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "T6: ";
    for (int i = 0; i < T6.value().size(); i++) {
        std::cout << T6.value()[i] << " ";
    }
    std::cout << std::endl;
}