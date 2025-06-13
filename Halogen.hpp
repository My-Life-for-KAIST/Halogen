#pragma once
#include "Tensor.hpp"
#include <vector>
#include <algorithm>
#include <numeric>
#include <filesystem>
#include <fstream>
#include <cmath>
namespace Halogen {
    class Node {
    public:
        Tensor<float> value;
        Tensor<float> grad;

        vector<Node*> in;
        vector<Node*> out;
        bool requiresGradient = true;

        virtual void forward() = 0;
        virtual void backward(Tensor<float>& grad, double lr) = 0;
        virtual ~Node() {}
    };

    class Graph {
    private:
        vector<Node*> topo;
        static thread_local Graph* _current;
    public:
        static Graph* current() { return _current; }
        static void setCurrent(Graph* g){ _current=g; }
        void add(Node *n) {
            topo.push_back(n);
        }
        Tensor<float>& forward() {
            for (Node* n : topo) n->forward();
            return topo.back()->value; // 그래프의 결과값
        }

        void backward() {
            if (topo.empty()) return;
            Node* loss = topo.back();
            loss->grad = Tensor<float>(loss->value.shape);
            for (auto& e : loss->grad.data) e = 1.0f;

            // 반대로 전파
            for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
                Node* n = *it;
                n->backward(n->grad /* upstream g */ , 0.0 /* learning rate는 무시*/);
            }
        }

        void zero_grad() {
            for (Node* n : topo) { // 전체 노드 순회
                if (n->requiresGradient) {
                    n->grad = Tensor<float>(n->value.shape); // gradient를 0으로 초기화
                }
            }
        }

        vector<Node*> parameters() const {
            vector<Node*> params;
            for (Node* n : topo) {
                if (n->requiresGradient && n->in.empty()) params.push_back(n);
            }
            return params;
        }
    };
    thread_local Graph* Graph::_current = nullptr;

    inline void ensure_grad(Node* n){
        if(n->grad.numel()==0) n->grad = Tensor<float>(n->value.getShape());
    }

    class Add: public Node {
    public:
        Add(Node *a, Node *b) {
            in = {a, b};
            Graph::current()->add(this);
        }
        void forward() override {
            value = in[0]->value + in[1]->value;
        }

        void backward(Tensor<float>& g, double lr=0) override {
            ensure_grad(in[0]);
            ensure_grad(in[1]);
            grad = grad + g;

            if (in[0]->requiresGradient)
                in[0]->grad = in[0]->grad + g;
            if (in[1]->requiresGradient)
                in[1]->grad = in[1]->grad + g;
        }
    };

    class Sub : public Node {
    public:
        Sub(Node* a, Node* b){
            in = {a, b};
            Graph::current()->add(this);
        }
        void forward() override { value = in[0]->value - in[1]->value; }
        void backward(Tensor<float>& g, double lr=0) override {
            ensure_grad(in[0]); ensure_grad(in[1]);
            in[0]->grad = in[0]->grad + g;
            in[1]->grad = in[1]->grad - g;
        }
    };

    class Mul : public Node {
    public:
        Mul(Node* a, Node *b) {
            in = {a, b};
            Graph::current()->add(this);
        }
        void forward() override { value = in[0]->value * in[1]->value; }
        void backward(Tensor<float>& g, double lr=0) override {
            ensure_grad(in[0]); ensure_grad(in[1]);
            in[0]->grad = in[0]->grad + g * in[1]->value;
            in[1]->grad = in[1]->grad + g * in[0]->value;
        }
    };

    class Div : public Node {
    public:
        Div(Node* a, Node* b){
            in = {a, b};
            Graph::current()->add(this);
        }
        void forward() override { value = in[0]->value / in[1]->value; }
        void backward(Tensor<float>& g, double lr=0) override {
            ensure_grad(in[0]); ensure_grad(in[1]);
            in[0]->grad = in[0]->grad + g / in[1]->value;                      // g / y
            in[1]->grad = in[1]->grad - g * in[0]->value / (in[1]->value*in[1]->value); // -g*x / y^2
        }
    };

    class MatrixMul : public Node {
    public:
        MatrixMul(Node* a, Node* b){
            in = {a, b};
            Graph::current()->add(this);
        }
        void forward() override {
            value = in[0]->value.matrixMul(in[1]->value);
        }
        void backward(Tensor<float>& g, double lr=0) override {
            ensure_grad(in[0]); ensure_grad(in[1]);
            // dA = g * B^T
            Tensor<float> BT = in[1]->value.transpose();
            in[0]->grad = in[0]->grad + g.matrixMul(BT);
            // dB = A^T * g
            in[1]->grad = in[1]->grad + in[0]->value.transpose().matrixMul(g);
        }
    };

    class ReLU : public Node {
    public:
        explicit ReLU(Node* x){
            in = { x };
            Graph::current()->add(this);
        }
        void forward() override {
            const auto& X = in[0]->value;
            value = Tensor<float>(X.shape);

            for (int i = 0; i < X.numel(); ++i)
                value.data[i] = max(0.0f, X.data[i]); // max(0,x)
        }
        void backward(Tensor<float>& g, double lr=0) override {
            const auto& X = in[0]->value;
            ensure_grad(in[0]);

            for (int i = 0; i < X.numel(); ++i) {
                float mask = X.data[i] > 0.0f ? 1.0f : 0.0f;
                in[0]->grad.data[i] += g.data[i] * mask; // g · 1{x>0}
            }
        }
    };

    class Sigmoid : public Node {
    public:
        explicit Sigmoid(Node* x){
            in={x};
            Graph::current()->add(this);
        }
        void forward() override {
            const auto& X = in[0]->value;
            value = Tensor<float>(X.shape);

            for (int i = 0; i < X.numel(); ++i) {
                float z = X.data[i];
                value.data[i] = 1.0f / (1.0f + exp(-z));      // σ(z)
            }
        }

        void backward(Tensor<float>& g, double lr=0) override {
            ensure_grad(in[0]);

            for (int i = 0; i < value.numel(); ++i) {
                float sigma = value.data[i];
                float deriv = sigma * (1.0f - sigma);
                in[0]->grad.data[i] += g.data[i] * deriv;
            }
        }
    };

    class Variable : public Node {
    public:
        Variable(const Tensor<float>& init, bool require_grad=false){
            value = init;
            requiresGradient = require_grad;
            if(require_grad) grad = Tensor<float>(init.shape);   // 0-tensor
            // Graph::current()->add(this);
        }
        // 순/역전파: 부모가 없으므로 아무 일도 안 함
        void forward() override {}
        void backward(Tensor<float>&, double) override {}
    };
    class Parameter : public Variable {
    public:
        Parameter(const Tensor<float>& init) : Variable(init, /*require_grad=*/true) {}
    };


    // -------- 옵티마이저 구현 -------------------------------------
    class Optimizer {
    public:
        virtual void step(const vector<Halogen::Node*>& parameters) = 0;
        virtual void zero_grad() {}
        virtual ~Optimizer() = default;
    };

    class SGD : public Optimizer {
    float lr;
    public:
        SGD(float _lr = 1e-2f) : lr(_lr) {}
        void step(const vector<Halogen::Node*>& parameters) {
            for (Halogen::Node* p: parameters) {
                p->value = p->value - p->grad * lr;
            }
        }
    };

    Tensor<float> read_mnist_images(const std::string& path, int limit=-1){

    }
}
