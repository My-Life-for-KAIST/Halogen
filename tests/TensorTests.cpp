#include "Halogen/Core/Tensor.h"
#include <cassert>

using Halogen::Tensor;

int main() {
    Tensor<int> tensor({1, 2, 3, 4, 5, 6}, {2, 3});

    auto validIndex = tensor.at({1, 2});
    assert(validIndex.has_value());
    assert(validIndex->get() == 6);

    auto outOfRangeIndex = tensor.at({1, 3});
    assert(!outOfRangeIndex.has_value());

    auto negativeIndex = tensor.at({-1, 1});
    assert(!negativeIndex.has_value());

    return 0;
}
