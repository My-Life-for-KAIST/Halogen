#pragma once
#include <vector>
#include <optional>
#include <functional>
#include <iostream>

// Halogen's Core Library : Really Super Important Tensor Lib.

// README!
/*
 * Thank you for using and contributing to Halogen Library.
 * While Contributing, using CLion(by Jetbrains) is "strongly" recommended.
 * I. Before Contributing
 *    1. non-const function must return OpTensorRef Type. ( equals to std::optional<std::reference_wrapper<Tensor<T>>> )
 *
 */
namespace Halogen {


    template <typename T>
    class Tensor {
        using OpTensorRef = std::optional<std::reference_wrapper<Tensor<T>>>;

        private:
            std::vector<int> strides; // 각 축의 인덱스가 1 증가할 때 실제 인덱스는 얼마나 증가하는가
            std::vector<T> data;

        public:
            std::vector<int> shape;
            // 스태틱 메서드
            static std::optional<Tensor> all_same(std::vector<int>& _shape, T value) {
                if (_shape.empty()) return std::nullopt;
                Tensor t; t.shape = std::move(_shape);

                int elementCount = 1;
                for (int v: t.shape) elementCount *= v;
                t.data.assign(elementCount, value);

                t.recompute_strides();
                return t;
            }

            static std::optional<Tensor> zeros(std::vector<int>& shape) {
                return all_same(shape, static_cast<T>(0));
            }

            // rvalue로 생성하는 경우
            static std::optional<Tensor> zeros(std::vector<int> shape) {
                return all_same(shape, static_cast<T>(0));
            }

            static std::optional<Tensor> identity(int dim, int N) {
                if (dim <= 0 || N <= 0) return std::nullopt;

                std::vector<int> _shape(dim, N);

                int total = 1;
                for (int i = 0; i < dim; ++i) total *= N;

                std::vector<T> _flat(total, static_cast<T>(0));
                const int step = (N-1 ? (total - 1) / (N-1) : 1);
                for (int k = 0; k < N; ++k) {
                    _flat[static_cast<size_t>(k) * step] = static_cast<T>(1);
                }

                return Tensor(std::move(_flat), std::move(_shape));
            }

            static std::optional<Tensor> arange(int stop) {
                if (stop <= 0) return Tensor(std::vector<T>{}, std::vector<int>{0});

                std::vector<int> _shape = {stop};
                std::vector<T> _flat(stop, static_cast<T>(0));
                for (int i = 0; i < stop; i++) _flat[i] = static_cast<T>(i);

                return Tensor(std::move(_flat), std::move(_shape));
            }

            static std::optional<Tensor> arange(int start, int stop, int step = 1) {
                if (step == 0) return std::nullopt;

                if ((step > 0 && start >= stop) || (step < 0 && start <= stop)) {
                    return Tensor(std::vector<T>{}, std::vector<int>{0});
                }

                std::vector<int> _shape = {(stop-start)/step};
                std::vector<T> _flat((stop-start)/step, static_cast<T>(0));
                int idx = 0;
                for (int i = start; i < stop; i += step) _flat[idx++] = static_cast<T>(i);

                return Tensor(std::move(_flat), std::move(_shape));
            }



            // -- 생성자 --
            // Initalizers
            Tensor() = default;
            Tensor(std::vector<T> flat, std::vector<int> shape_):
            data(std::move(flat)), shape(std::move(shape_)) {
                strides.assign(shape.size(), 0);
                int acc = 1;
                for (int d = shape.size() - 1; d >= 0; --d) {
                    strides[d] = acc;
                    acc *= shape[d];
                }
            }

            void recompute_strides() {
                strides.assign(shape.size(), 0);
                int acc = 1;
                for (int d = shape.size() - 1; d >= 0; --d) {
                    strides[d] = acc;
                    acc *= shape[d];
                }
            }

            // -- 연산자 오버로딩 --

            /**
             * @brief Tensor의 위치에 접근하는 배열 첨자 연산
             * @return 해당 위치의 레퍼런스 (unsafe)
             */
            template <class... Is>
            T& operator[](Is... is) {
                const std::vector<int> idx{ static_cast<int>(is)... };
                return data[offset(idx)];
            }

            Tensor operator-() {
                Tensor res = *this;
                res.map([] (T& v) {
                    v *= -1;
                });
                return res;
            }

            Tensor operator+(T& other) {
                Tensor res = *this;
                res.map([other] (T& v) {
                    v += other;
                });
                return res;
            }

            Tensor operator-(T& other) {
                Tensor res = *this;
                res.map([other] (T& v) {
                    v -= other;
                });
                return res;
            }

            Tensor operator*(T& other) {
                Tensor res = *this;
                res.map([other] (T& v) {
                    v *= other;
                });
                return res;
            }

            OpTensorRef add(int other) {
                for (auto& v : data) v += static_cast<T>(other);
                return *this;
            }

            OpTensorRef add(const Tensor& other) {
                if (size() != other.size()) return std::nullopt;
                for (int i = 0; i < size(); ++i) data[i] += other.data[i];
                return *this;
            }

            OpTensorRef sub(int other) {
                for (auto& v : data) v -= static_cast<T>(other);
                return *this;
            }

            OpTensorRef sub(const Tensor& other) {
                if (size() != other.size()) return std::nullopt;
                for (int i = 0; i < size(); ++i) data[i] -= other.data[i];
                return *this;
            }

            OpTensorRef mul(int other) {
                for (auto& v : data) v *= static_cast<T>(other);
                return *this;
            }

            OpTensorRef mul(const Tensor& other) {
                if (size() != other.size()) return std::nullopt;
                for (int i = 0; i < size(); ++i) data[i] *= static_cast<T>(other.data[i]);
                return *this;
            }

            OpTensorRef matmul(const Tensor& other) {
                if (ndim() != 2 || other.ndim() != 2) return std::nullopt;

                const int m = shape[0];
                const int n = other.shape[1];
                if (shape[1] != other.shape[0]) return std::nullopt; // (m×k) * (k×n)

                std::vector<T> out(m * n, T{});

                for (int i = 0; i < m; i++) {
                    const int a_row = i * shape[1];
                    for (int j = 0; j < n; j++) {
                        T acc{};
                        for (int t = 0; t < shape[1]; ++t) acc += data[a_row + t] * other.data[t*n + j];
                        out[i*n + j] = acc;
                    }
                }

                // this 업데이트
                data = std::move(out);
                shape = {m, n};
                recompute_strides();
                return *this;
            }

            // -- 기본적 텐서 속성 --

            [[nodiscard]] int ndim() const {
                return shape.size();
            }

            /**
             * @brief 해당 축의 크기 반환
             * @param axis 축
             * @return 해당 축의 크기
             */
            [[nodiscard]] std::optional<int> dim(int axis) const {
                if (axis < 0 || axis >= shape.size()) {
                    return std::nullopt;
                }
                return shape[axis];
            }

            /**
             * @brief 모든 데이터의 개수 반환
             * @return 모든 데이터의 개수 (int)
             */
            [[nodiscard]] int size() const {
                return data.size();
            }

            /**
             * @brief N차원 축 인덱스를 1차원 인덱스(내부 데이터)로 변환
             * @param idx N차원 축 인덱스
             * @return 변환한 1차원 인덱스를 반환
             */
            [[nodiscard]] int offset(const std::vector<int>& idx) const {
                int ofs = 0;
                for (int i = 0; i < shape.size(); i++) {
                    ofs += idx[i] * strides[i];
                }
                return ofs;
            }

            /**
             * @brief 배열 첨자 접근에 범위 체크 추가 (safe access)
             * @param idx 접근할 위치의 인덱스
             * @return 해당 위치의 data의 레퍼런스를 reference_wrapper로 감싸 전달
             */
            [[nodiscard]] std::optional<std::reference_wrapper<T>> at(const std::vector<int>& idx) {
                // range check
                if (idx.size() != shape.size()) return std::nullopt;
                for (int i = 0; i < shape.size(); i++) {
                    if (idx[i] >= shape.size() || idx[i] < 0) {
                        return std::nullopt;
                    }
                }
                return data[offset(idx)];
            }

            // ------- 텐서 변형 함수 -------

            /**
             * @brief 텐서의 차원 및 축별 크기 변경 (reshape)
             * @param newShape 바꿀 텐서 축별 크기 (shape)
             * @return 인수로 들어온 크기가 맞지 않으면 nullopt 반환, 잘 실행되면 true 반환
             */
            OpTensorRef reshape(const std::vector<int>& newShape) {
                if (newShape.empty()) return std::nullopt;

                int elementCount = 1;
                for (const int x: newShape) elementCount *= x;

                if (elementCount != data.size()) return std::nullopt;

                shape = newShape;
                recompute_strides();
                return *this;
            }

            // ------- 텐서 연산 함수 -------
            /**
                @brief 요소에 일괄적으로 연산 적용
                @param func 각 요소에 적용할 함수 (텐서의 한 요소를 매개변수로 받고 결과값을 리턴)
             **/
            OpTensorRef apply(std::function<T(T)>& func) {
                for (auto &t: data) {
                    t = func(t);
                }
                return *this;
            }

            /**
                @brief 요소에 일괄적으로 연산 적용
                @param func 각 요소에 적용할 함수 (텐서의 한 요소의 레퍼런스를 매개변수로 받고 그 안에서 레퍼런스에 대입, 리턴 값 없음)
             **/
            OpTensorRef map(std::function<void(T&)> func) {
                for (auto& t: data) {
                    func(t);
                }
                return *this;
            }

            // ------- 텐서 검사 -------
            /**
               @brief 모든 요소에 대해 func를 적용했을 때 true이면 true 반환
               @param func 각 요소에 적용할 함수 (텐서의 한 요소를 매개변수로 받아 bool 반환)
             **/
            bool all(std::function<bool(T)> func) const {
                for (const auto &t: data) {
                    if (!func(t)) return false;
                }
                return true;
            }

            /**
               @brief 모든 요소에 대해 func를 적용했을 때 하나라도 true이면 true 반환
               @param func 각 요소에 적용할 함수 (텐서의 한 요소를 매개변수로 받아 bool 반환)
             **/
            bool any(std::function<bool(T)> func) const {
                for (const auto &t: data) {
                    if (func(t)) return true;
                }
                return false;
            }


    };
}
