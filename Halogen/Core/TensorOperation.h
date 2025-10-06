#pragma once
#include <optional>
#include "Tensor.h"

namespace Halogen::TensorOperation {
    template <typename T>
    using OpTensorRef = std::optional<std::reference_wrapper<Tensor<T>>>;

    template<class T, class F>
    concept Stage =
    std::invocable<F, Tensor<T>&> && // F는 호출 가능
    std::same_as<std::invoke_result_t<F, Tensor<T>&>, OpTensorRef<T>>; // 반환형이 OpTensorRef<T>

    template<class T>
    inline OpTensorRef<T> bind(Tensor<T>& x) noexcept {
        return std::ref(x);
    }

    template<class T, class F> requires Stage<T, F>
    inline OpTensorRef<T> operator>>(OpTensorRef<T>&& opt, F&& f) {
        if (!opt) return std::nullopt;
        return std::invoke(std::forward<F>(f), opt->get());
    }

    // -----------------------------------------------------------------
    inline auto reshape = [](std::vector<int> shape) {
        return [shape = std::move(shape)](auto& t) {
            return t.reshape(shape);
        };
    };

    inline auto add(int other) {
        return [other](auto& t) { return t.add(other); };
    }

    template<class T>
    inline auto add(const Tensor<T>& other) {
        return [&other](auto& t) { return t.add(other); };
    }

    inline auto sub(int other){
        return [other](auto& t) { return t.sub(other); };
    }

    template<class T>
    inline auto sub(const Tensor<T>& other) {
        return [&other](auto& t) { return t.sub(other); };
    }

    inline auto mul(int other) {
        return [other](auto& t) { return t.mul(other); };
    }

    template<class T>
    inline auto mul(const Tensor<T>& other) {
        return [&other](auto& t) { return t.mul(other); };
    }

    template<class T>
    inline auto matmul(const Tensor<T>& other) {
        return [&other](auto& t) { return t.matmul(other); };
    }
    // -----------------------------------------------------------------

}
