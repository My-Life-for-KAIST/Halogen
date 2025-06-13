#pragma once
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <stdexcept>
using namespace std;

namespace Halogen {
    template<typename T>
    class Tensor {
    private:

        vector<int> strides; // 각 축의 인덱스가 1 증가할 때 실제 인덱스는 얼마나 증가하는가
        void recursiveAdd(Tensor& out, const Tensor& other, vector<int>& idx, int axis) {
            if (axis == shape.size()) {
                int of = offset(idx);
                out.data[of] = data[of] + other.data[of];
                return;
            }
            for (int i = 0; i < shape[axis]; i++) {
                idx[axis] = i;
                recursiveAdd(out, other, idx, axis+1);
            }
        }
        void recursiveSubtract(Tensor& out, const Tensor& other, vector<int>& idx, int axis) {
            if (axis == shape.size()) {
                int of = offset(idx);
                out.data[of] = data[of] - other.data[of];
                return;
            }
            for (int i = 0; i < shape[axis]; i++) {
                idx[axis] = i;
                recursiveSubtract(out, other, idx, axis+1);
            }
        }
        void recursiveMultiply(Tensor& out, const Tensor& other, vector<int>& idx, int axis) const {
            if (axis == shape.size()) {
                int of = offset(idx);
                out.data[of] = data[of] * other.data[of];
                return;
            }
            for (int i = 0; i < shape[axis]; i++) {
                idx[axis] = i;
                recursiveMultiply(out, other, idx, axis+1);
            }
        }
        void recursiveDivide(Tensor& out, const Tensor& other, vector<int>& idx, int axis) const {
            if (axis == shape.size()) {
                int of = offset(idx);
                out.data[of] = data[of] / other.data[of];
                return;
            }
            for (int i = 0; i < shape[axis]; i++) {
                idx[axis] = i;
                recursiveDivide(out, other, idx, axis+1);
            }
        }
    public:
        Tensor() {};
        vector<T> data; // 1차원 데이터
        vector<int> shape; // 축 별 크기
        int ndim() const { return shape.size(); }
        int dim(int axis) { return shape.at(axis); }
        int numel() const {
            return accumulate(shape.begin(), shape.end(), 1, multiplies<int>());
        }
        vector<int> getShape() { return shape; }
        Tensor(const vector<int> &_shape): shape(_shape) {
            strides.resize(_shape.size());
            int acc = 1;
            for (int i = _shape.size() - 1; i>=0; i--) {
                strides[i] = acc;
                acc *= _shape[i];
            }
            data.assign(acc, 0);
        }
        Tensor(const vector<T>& _data, const vector<int>& _shape, const vector<int>& _strides) {
            data = _data;
            shape = _shape;
            strides = _strides;
        }
        int offset(const vector<int>& idx) const{
            int ofs = 0;
            for (int i = 0; i < shape.size(); i++) {
                ofs += idx[i] * strides[i];
            }
            return ofs;
        }

        T& operator() (const vector<int>& idx) {
            return data[offset(idx)];
        }
        const T& operator()(const vector<int>& idx) const { return data[offset(idx)]; }

        T& operator() (const initializer_list<int>& idx) {
            int ofs = 0;
            auto it = idx.begin();
            for (int i = 0; i < shape.size(); i++, it++) {
                ofs += *it * strides[i];
            }
            return data[ofs];
        }
        const T& operator() (const initializer_list<int>& idx) const{
            int ofs = 0;
            auto it = idx.begin();
            for (int i = 0; i < shape.size(); i++, it++) {
                ofs += *it * strides[i];
            }
            return data[ofs];
        }

        Tensor operator+(const Tensor& other) {
            if (shape != other.shape){
                throw "Shape Mismatch";
            }
            Tensor result(shape);
            vector<int> idx(shape.size(), 0);
            recursiveAdd(result, other, idx, 0);
            return result;
        }

        Tensor operator+(const T& scalar) {
            Tensor result(*this);
            for (auto& E: result.data) E += scalar;
            return result;
        }

        Tensor operator-() {
            Tensor result(*this);
            for (auto& E: result.data) E *= -1;
            return result;
        }
        Tensor operator-(const T& scalar) {
            Tensor result(*this);
            for (auto& E: result.data) E -= scalar;
            return result;
        }
        Tensor operator-(const Tensor& other) {
            if (shape != other.shape){
                throw "Shape Mismatch";
            }
            Tensor result(shape);

            vector<int> idx(shape.size(), 0);
            recursiveSubtract(result, other, idx, 0);
            return result;
        }

        Tensor operator*(const T& scalar) {
            Tensor result(*this);
            for (auto& E: result.data) E *= scalar;
            return result;
        }
        Tensor operator*(const Tensor& other) const{
            if (shape != other.shape){
                throw "Shape Mismatch";
            }
            Tensor result(shape);

            vector<int> idx(shape.size(), 0);
            recursiveMultiply(result, other, idx, 0);
            return result;
        }
        Tensor operator/(const T& scalar) {
            Tensor result(*this);
            for (auto& E: result.data) E /= scalar;
            return result;
        }
        Tensor operator/(const Tensor& other) const {
            if (shape != other.shape){
                throw "Shape Mismatch";
            }
            Tensor result(shape);

            vector<int> idx(shape.size(), 0);
            recursiveDivide(result, other, idx, 0);
            return result;
        }
        Tensor matmul2d(const Tensor& B) const{
            if(ndim() != 2 || B.ndim() != 2 || shape[1] != B.shape[0]) throw "Invalid Dimension in MatrixMultiplying";


            int m = shape[0], k=shape[1], n=B.shape[1];
            Tensor C({m,n});
            for(int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    T acc = 0;
                    for (int p = 0; p < k; p++) {
                        acc += (*this)({i, p}) * B({p, j});
                    }
                    C({i, j}) = acc;
                }
            }
            return C;
        }
        Tensor matrixMul(Tensor& B) {
            if(shape.back() != B.shape[B.ndim()-2]) throw "inner dim mismatch";
            if(ndim() == 2 && B.ndim() == 2) return matmul2d(B);
            if(ndim()==3 && B.ndim() == 3){
                int Bn=shape[0], Bm=B.shape[0];
                if(Bn!=Bm) throw std::invalid_argument("batch dim mismatch"); int m=shape[1],k=shape[2],n=B.shape[2];
                Tensor out({Bn,m,n});
                for(int b = 0; b < Bn; b++){
                    for(int i = 0; i < m; i++){
                        for(int j = 0; j < n; j++){
                            T acc = 0;
                            for(int p = 0; p < k; p++) {
                                acc += (*this)({b,i,p}) * B({b,p,j});
                            }
                            out({b,i,j}) = acc;
                        }
                    }
                }
                return out;
            }
            return Tensor();
        }

        Tensor transpose(int a=-2, int b=-1) const {
            if (ndim()<2) throw std::invalid_argument("transpose: ndim<2");
            int na=ndim(); if(a<0) a+=na; if(b<0) b+=na;
            vector<int> tsh = shape;
            vector<int> st = strides;
            swap(tsh[a], tsh[b]);
            swap(st[a], st[b]);
            return Tensor(data, tsh, st);
        }
        Tensor reshape(const vector<int>& new_shape) const {
            if (numel() != accumulate(new_shape.begin(), new_shape.end(), 1, multiplies<int>()))
                throw invalid_argument("reshape: element count mismatch");
            vector<int> st(new_shape.size());
            int acc = 1;
            for(int i = new_shape.size() - 1; i >= 0; i--){
                st[i]=acc;
                acc*=new_shape[i];
            }
            return Tensor(data, new_shape, st);
        }
        Tensor squeeze(int axis = -1) const {
            vector<int> new_shape;
            for (int i = 0; i < ndim(); i++) {
                if ((axis == -1 && shape[i] == 1) || i == axis) {
                    if(shape[i]!=1) throw "squeeze: axis dim not 1";
                    continue;
                }
                new_shape.push_back(shape[i]);
            }
            if (new_shape.empty()) new_shape.push_back(1);
            return reshape(new_shape);
        }
        Tensor applyReLU() {
            Tensor out(shape);
            for (int i = 0; i < data.size(); ++i)
                out.data[i] = max(0.0f, data[i]);
            return out;
        }
        Tensor sqrt() const {
            Tensor out(shape);
            for (size_t i=0;i<numel();++i) out.data[i] = sqrt(data[i]);
            return out;
        }
    };

}
