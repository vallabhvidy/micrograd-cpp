#include "micrograd.h"
#include <iostream>
#include <math.h>
#include <set>
#include <cstdlib>
#include <vector>
using namespace std;

Value::Value(float _data, vector<Value*> children, char _op) {
    data = _data;
    prev = children;
    op = _op;
    grad = 0;
}

float Value::get_data() {
    return data;
}

Value operator+(Value& a, Value& b) {
    Value c(a.data + b.data, {&a, &b}, '+');
    return c;
}

// Value operator+(float a, Value& b) {
//     Value c(a);
//     return (c + b);
// }

// Value operator+(Value& a, float b) {
//     Value c(b);
//     return (c + a);
// }

Value operator*(Value& a, Value& b) {
    Value c(a.data * b.data, {&a, &b}, '*');
    return c;
}

// Value operator*(float a, Value& b) {
//     Value c(a);
//     return (c * b);
// }

// Value operator*(Value& a, float b) {
//     Value c(b);
//     return (c * a);
// }

Value tanh(Value& a) {
    float n = (exp(a.data) - exp(-a.data)) / (exp(a.data) + exp(-a.data));
    Value c(n, {&a}, 't');
    return c;
}

Value exp(Value& a) {
    float n = exp(a.data);
    Value c(n, {&a}, 'e');
    return c;
}

Value operator-(Value& a, Value& b) {
    Value c(a.data - b.data, {&a, &b}, '-');
    return c;
}

// Value operator-(float a, Value& b) {
//     Value c(a);
//     return (c - b);
// }

// Value operator-(Value& a, float b) {
//     Value c(b);
//     return (a - c);
// }

Value operator/(Value& a, Value& b) {
    Value c(a.data / b.data, {&a, &b}, '/');
    return c;
}

Value relu(Value& a) {
    Value c((a.data > 0 ? a.data : 0.01 * a.data), {&a}, 'r');
    return c;
}

Value sqrdiff(Value& a, Value& b) {
    Value c((float)pow((a.data - b.data), 2), {&a, &b}, 's');
    return c;
}

void operator+=(Value& a, Value& b) {
    a.op = 'p';
    a.data += b.data;
    a.prev.push_back(&b);
}

// Value operator/(float a, Value& b) {
//     Value c(a);
//     return (c / b);
// }

// Value operator/(Value& a, float b) {
//     Value c(b);
//     return (a / c);
// }

void Value::backward() {
    vector<Value*> topo;
    set<Value*> visited;

    build_topo(this, topo, visited);

    grad = 1;

    for (int i = topo.size()-1; i >= 0; i--) {
        topo[i]->_backward();
    }
}

void Value::build_topo(Value* v, vector<Value*>& topo, set<Value*>& visited) {
    if (v == nullptr) return;
    if (visited.find(v) == visited.end()) {
        visited.insert(v);
        // cout << v->data << endl;
        for (Value* child: v->prev) {
            build_topo(child, topo, visited);
        }
        topo.push_back(v);
    }
}

void Value::zero_grad() {
    grad = 0;
}

Neuron::Neuron(int nin) {
    w = vector<Value*>(nin);
    for (auto& weight: w) weight = new Value();
}

Value* Neuron::predict(vector<Value*> in, bool lin) {
    if (in.size() != w.size()) {
        cout << "Input size does not match" << endl;
        return nullptr;
    }
    Value* sum = new Value(0);
    sum = new Value(*sum + *b);
    auto k = w;
    for (int i = 0; i < w.size(); i++) {
        Value* val = in[i];
        Value* mul = new Value((*w[i]) * (*val));
        sum = new Value((*sum) + (*mul));
    }

    if (!lin) sum = new Value(tanh(*sum));

    return sum;
}

Layer::Layer(int nin, int nout) {
    neurons.resize(nout, Neuron(nin));
}

vector<Value*> Layer::predict(vector<Value*> in, bool lin) {
    vector<Value*> act;
    for (Neuron n: neurons) {
        act.push_back(n.predict(in, lin));
    }

    return act;
}

MLP::MLP(int nin, vector<int> nout) {
    layers.push_back(Layer(nin, nout[0]));
    for (int i = 0; i < nout.size()-1; i++) {
        layers.push_back(Layer(nout[i], nout[i+1]));
    }
}

vector<Value*> MLP::predict(vector<float> in, bool lin) {
    vector<Value*> x;
    for (float i: in) x.push_back(new Value(i));
    for (int i = 0; i < layers.size()-1; i++) {
        x = layers[i].predict(x, lin);
    }

    x = layers.back().predict(x, lin);

    return x;
}

void Neuron::zero_grad() {
    for (Value* weight: w) weight->zero_grad();
    b->zero_grad();
}

void Layer::zero_grad() {
    for (Neuron n: neurons) n.zero_grad();
}

void MLP::zero_grad() {
    for (Layer l: layers) l.zero_grad();
}

vector<Value*> Neuron::get_params() {
    vector<Value*> params = w;
    params.push_back(b);
    return params;
}

vector<Value*> Layer::get_params() {
    vector<Value*> params;
    for (Neuron n: neurons) {
        vector<Value*> n_params = n.get_params();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}

Value operator^(Value& a, int b) {
    Value c(pow(a.data, b), {&a}, '^');
    c.power = b;
    return c;
}

vector<Value*> MLP::get_params() {
    vector<Value*> params;
    for (Layer l: layers) {
        vector<Value*> l_params = l.get_params();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}

void Value::_backward() {
    if (op == '+') {
        prev[0]->grad += grad;
        prev[1]->grad += grad;
    } else if (op == '*') {
        prev[0]->grad += prev[1]->data * grad;
        prev[1]->grad += prev[0]->data * grad;
    } else if (op == 't') {
        prev[0]->grad += (1 - data * data) * grad;
    } else if (op == 'e') {
        prev[0]->grad += (data * grad);
    } else if (op == '/') {
        prev[0]->grad += grad / prev[1]->data;
        prev[1]->grad += (-grad * prev[0]->data) / (prev[1]->data * prev[1]->data);
    } else if (op == '-') {
        prev[0]->grad += grad;
        prev[1]->grad += -grad;
    } else if (op == '^') {
        prev[0]->grad += power * (pow(prev[0]->data, power-1)) * grad;
    } else if (op == 'r') {
        prev[0]->grad += (prev[0]->data > 0 ? 1 : 0.01) * grad;
    } else if (op == 's') {
        prev[0]->grad += 2 * (prev[0]->data - prev[1]->data) * grad;
        prev[1]->grad -= 2 * (prev[0]->data - prev[1]->data) * grad;
    } else if (op == 'p') {
        for (auto& p: prev) {
            p->grad += grad;
        } 
    }
}

void Value::moddata(float lr) {
    data += ((-lr) * grad);
}

void MLP::train(vector<vector<float>> xs, vector<vector<float>> ys, int epoch) {
    for (int i = 0; i < epoch; i++) {
        vector<Value*> deletes;
        Value* loss = new Value(0.00);
        deletes.push_back(loss);
        vector<Value*> ypred;
        for (int j = 0; j < ys.size(); j++) {
            ypred = predict(xs[j]);
            deletes.insert(deletes.end(), ypred.begin(), ypred.end());
            for (int k = 0; k < ypred.size(); k++) {
                Value* y = new Value(ys[j][k]);
                Value* sdiff = new Value(sqrdiff(*ypred[k], *y));
                *loss += *sdiff;
                deletes.push_back(sdiff);
            }
        }

        zero_grad();

        loss->backward();

        float lr = 0.001;

        if (loss->get_data() > 0.01) {
            lr = 0.01;
        } else if (loss->get_data() > 0.005) {
            lr = 0.007;
        } else if (loss->get_data() > 0.001) {
            lr = 0.004;
        } else {
            lr = 0.002;
        }

        for (Value* p: get_params()) {
            p->moddata(0.01);
        }

        if ((i+1) % 1000 == 0) cout << loss->get_data() << " " << i+1 << endl;

        for (auto ptr: deletes) delete ptr;
    }

    
}

float f(float x) {
    // modify this function accordingly
    // and make sure it normalized to [-1, 1]
    return 2*(float)pow(2, sin(x))-3;
}

int main() {
    // modify the structure of the net here 
    MLP n(1, {16, 8, 1});
    
    vector<vector<float>> X;
    vector<vector<float>> Y;
    for (float i = 0; i <= 1; i += 0.1) {
        // try to predict different mathematical functions
        X.push_back({i});
        // Y.push_back({sin(i)});
        // Y.push_back({(float)log(1 + pow(i, 2))});
        Y.push_back({f(i)});
    }
    n.train(X, Y, 25000);

    // predict using the mlp.predict method and print the denormalized output
    auto pred = n.predict({0.55});
    cout << (pred[0]->get_data()+3) / 2 << endl;

    // backward and forward pass for a single iteration

    // vector<Value*> y = n.predict({1.0});
    // cout << y[0]->get_data() << endl;
    // Value ye = 1.00;
    // Value diff = *y[0] - ye;
    // Value loss = diff ^ 2;
    // loss.backward();
    // vector<Value*> params = n.get_params();
    // for (Value* p: n.get_params()) {
    //     p->moddata(0.01);
    // }
    // y = n.predict({1.0});
    // cout << y[0]->get_data() << endl;

}
