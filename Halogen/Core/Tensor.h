#pragma once
#include <vector>
#include <optional>
#include <functional>

// Halogen's Core Library : Really Super Important Tensor Lib.

// README!
/*
 * Thank you for using and contributing to Halogen Library.
 * While Contributing, using CLion(by Jetbrains) is "strongly" recommended.
 * I. Before Contributing
 *    1. non-const function must return OpTensorRef Type. ( equals to sstd::optional<std::reference_wrapper<Tensor<T>>> )
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
            // 스태틱 메서드


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

            // -- 기본적 텐서 속성 --
            std::vector<int> shape;

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
                strides.assign(shape.size(), 0);
                int acc = 1;
                for (int d = shape.size() - 1; d >= 0; --d) {
                    strides[d] = acc;
                    acc *= shape[d];
                }
                return *this;
            }

            // ------- 텐서 연산 함수 -------
            /**
                @brief 요소에 일괄적으로 연산 적용
                @param func 각 요소에 적용할 함수 (텐서의 한 요소를 매개변수로 받고 결과값을 리턴)
             **/
            OpTensorRef apply(std::function<T(T)> func) {
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
