#pragma once
#include <optional>

#include "Tensor.h"

namespace Halogen::TensorOperation {
    template <typename T>
    using OpTensorRef = std::optional<std::reference_wrapper<Tensor<T>>>;

    template<class T>
    inline OpTensorRef<T> bind(Tensor<T>& x) noexcept {
        return std::ref(x);
    }

    template<typename T, typename F>
    inline OpTensorRef<T> operator>>(OpTensorRef<T>&& opt, F&& f) {
        if (!opt) return std::nullopt;
        return std::invoke(std::forward<F>(f), opt->get());
    }

    inline auto reshape(std::vector<int> s) {
        return [s = std::move(s)](auto& t) -> decltype(t.reshape(s)) {
            return t.reshape(s);
        };
    }

    template<typename S>
    inline auto add_(S s) {
        return [=](auto& t) -> decltype(t.map([&](auto& v){ v += s; })) {
            t.map([&](auto& v){ v = static_cast<std::decay_t<decltype(v)>>(v + s); });
            return t;
        };
    }

    template<typename S>
    struct Add {
        S s;
        template<typename U>
        OpTensorRef<U> operator()(Tensor<U>& t) const {
            t.map([&](U& v){ v = static_cast<U>(v + s); });
            return t;
        }
    };

    template<typename S> inline Add<S> add(S s){ return Add<S>{s}; }
}
