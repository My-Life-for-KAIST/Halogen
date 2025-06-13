#include <iostream>
#include "Halogen.hpp"

#define endl '\n';

using namespace Halogen;

int main() {
    // 계산 그래프 활성화
    Graph g;
    Graph::setCurrent(&g);

    // 입력 & 파라미터 생성
    auto x  = new Variable(Tensor<float>({32, 784}));  // batch 32
    auto W1 = new Parameter(Tensor<float>({784, 128}));
    auto b1 = new Parameter(Tensor<float>({128}));

    // 그래프를 연결
    auto z1 = new MatrixMul(x, W1);          // x @ W1
    auto a1 = new Add(z1, b1);            // + b1
    auto h1 = new ReLU(a1);               // ReLU

    auto params = g.parameters();
    SGD optimizer(1e-3);
    for (int i = 0; i < 32; i++) {
        g.zero_grad();
        g.forward();
        g.backward();
        optimizer.step(params);
    }
}
