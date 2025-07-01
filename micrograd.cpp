#include "micrograd.h"
#include <iostream>
#include <math.h>
#include <set>
#include <cstdlib>
#include <vector>
#include <algorithm>
using namespace std;

Value::Value(float _data, vector<Value *> children, char _op, bool _temp)
{
    data = _data;
    prev = children;
    op = _op;
    grad = 0;
    temp = _temp;
}

float Value::get_data() { return data; }

float Value::get_grad() { return grad; }

Value *operator+(const Value &a, const Value &b)
{
    Value *c = new Value(a.data + b.data, {(Value *)&a, (Value *)&b}, '+');
    return c;
}

Value *operator*(const Value &a, const Value &b)
{
    Value *c = new Value(a.data * b.data, {(Value *)&a, (Value *)&b}, '*');
    return c;
}

Value *tanh(const Value &a)
{
    float n = (exp(a.data) - exp(-a.data)) / (exp(a.data) + exp(-a.data));
    Value *c = new Value(n, {(Value *)&a}, 't');
    return c;
}

Value *exp(const Value &a)
{
    float n = exp(a.data);
    Value *c = new Value(n, {(Value *)&a}, 'e');
    return c;
}

Value *operator-(const Value &a, const Value &b)
{
    Value *c = new Value(a.data - b.data, {(Value *)&a, (Value *)&b}, '-');
    return c;
}

Value *operator/(const Value &a, const Value &b)
{
    Value *c = new Value(a.data / b.data, {(Value *)&a, (Value *)&b}, '/');
    return c;
}

Value *relu(const Value &a)
{
    Value *c = new Value((a.data > 0 ? a.data : 0.01 * a.data), {(Value *)&a}, 'r');
    return c;
}

Value *sqrdiff(const Value &a, const Value &b)
{
    Value *c = new Value((float)pow((a.data - b.data), 2), {(Value *)&a, (Value *)&b}, 's');
    return c;
}

void operator+=(Value &a, const Value &b)
{
    a.op = 'p';
    a.data += b.data;
    a.prev.push_back((Value *)&b);
}

Value *operator^(Value &a, const int b)
{
    Value *c = new Value(pow(a.data, b), {(Value *)&a}, '^');
    c->power = b;
    return c;
}

void Value::moddata(float lr)
{
    data += ((-lr) * grad);
}

void Value::_backward()
{
    if (op == '+')
    {
        prev[0]->grad += grad;
        prev[1]->grad += grad;
    }
    else if (op == '*')
    {
        prev[0]->grad += prev[1]->data * grad;
        prev[1]->grad += prev[0]->data * grad;
    }
    else if (op == 't')
    {
        prev[0]->grad += (1 - data * data) * grad;
    }
    else if (op == 'e')
    {
        prev[0]->grad += (data * grad);
    }
    else if (op == '/')
    {
        prev[0]->grad += grad / prev[1]->data;
        prev[1]->grad += (-grad * prev[0]->data) / (prev[1]->data * prev[1]->data);
    }
    else if (op == '-')
    {
        prev[0]->grad += grad;
        prev[1]->grad += -grad;
    }
    else if (op == '^')
    {
        prev[0]->grad += power * (pow(prev[0]->data, power - 1)) * grad;
    }
    else if (op == 'r')
    {
        prev[0]->grad += (prev[0]->data > 0 ? 1 : 0.01) * grad;
    }
    else if (op == 's')
    {
        prev[0]->grad += 2 * (prev[0]->data - prev[1]->data) * grad;
        prev[1]->grad -= 2 * (prev[0]->data - prev[1]->data) * grad;
    }
    else if (op == 'p')
    {
        for (auto &p : prev)
        {
            p->grad += grad;
        }
    }
}

void Value::backward()
{
    vector<Value *> topo;
    unordered_set<Value *> visited;

    build_topo(this, topo, visited);

    grad = 1;

    for (int i = topo.size() - 1; i >= 0; i--)
        topo[i]->_backward();
}

void Value::build_topo(Value *v, vector<Value *> &topo, unordered_set<Value *> &visited)
{
    if (v == nullptr)
        return;

    if (visited.find(v) == visited.end())
    {
        visited.insert(v);

        for (Value *child : v->prev)
            build_topo(child, topo, visited);

        topo.push_back(v);
    }
}

void Value::zero_grad()
{
    grad = 0;
}

void Value::deleteChildren()
{
    unordered_set<Value *> visited;
    deleteChildren(this, visited);
}

void Value::deleteChildren(Value *ptr, unordered_set<Value *> &visited)
{
    for (auto child : ptr->prev)
    {
        if (visited.find(child) == visited.end() && child->temp)
        {
            visited.insert(child);
            deleteChildren(child, visited);
            delete child;
        }
    }
}

Neuron::Neuron(int nin)
{
    w = vector<Value *>(nin);
    b = new Value();
    b->temp = false;
    for (auto &weight : w)
    {
        weight = new Value();
        weight->temp = false;
    }
}

Layer::Layer(int nin, int nout)
{
    neurons.resize(nout, Neuron(nin));
}

MLP::MLP(int nin, vector<int> nout)
{
    layers.push_back(Layer(nin, nout[0]));
    for (int i = 0; i < nout.size() - 1; i++)
        layers.push_back(Layer(nout[i], nout[i + 1]));
}

vector<Value *> Neuron::get_params()
{
    vector<Value *> params = w;
    params.push_back(b);
    return params;
}

vector<Value *> Layer::get_params()
{
    vector<Value *> params;
    for (Neuron &n : neurons)
    {
        vector<Value *> n_params = n.get_params();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}

vector<Value *> MLP::get_params()
{
    vector<Value *> params;
    for (Layer &l : layers)
    {
        vector<Value *> l_params = l.get_params();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}

void Neuron::zero_grad()
{
    for (Value *weight : w)
        weight->zero_grad();
    b->zero_grad();
}

void Layer::zero_grad()
{
    for (Neuron &n : neurons)
        n.zero_grad();
}

void MLP::zero_grad()
{
    for (Layer &l : layers)
        l.zero_grad();
}

Value *Neuron::predict(vector<Value *> in, bool lin)
{
    if (in.size() != w.size())
    {
        cout << "Input size does not match" << endl;
        return nullptr;
    }
    Value *sum = new Value(0);
    *sum += *b;
    auto k = w;
    for (int i = 0; i < w.size(); i++)
    {
        Value *val = in[i];
        Value *mul = (*w[i]) * (*val);
        *sum += *mul;
    }

    if (!lin)
        sum = tanh(*sum);

    return sum;
}

vector<Value *> Layer::predict(vector<Value *> in, bool lin)
{
    vector<Value *> act;
    for (Neuron n : neurons)
        act.push_back(n.predict(in, lin));

    return act;
}

vector<Value *> MLP::predict(vector<float> in)
{
    vector<Value *> x;
    for (float i : in)
        x.push_back(new Value(i));
    for (int i = 0; i < layers.size() - 1; i++)
        x = layers[i].predict(x);

    x = layers.back().predict(x, true);

    return x;
}

void MLP::train(vector<vector<float>> xs, vector<vector<float>> ys, int epoch)
{
    cout << "Loss:- " << endl;
    float lr = 0.0001;
    float prev = 0;
    for (int i = 0; i < epoch; i++)
    {
        Value *loss = new Value(0.00);
        vector<Value *> ypred;
        for (int j = 0; j < ys.size(); j++)
        {
            ypred = predict(xs[j]);
            for (int k = 0; k < ypred.size(); k++)
            {
                Value *y = new Value(ys[j][k]);
                Value *sdiff = sqrdiff(*ypred[k], *y);
                *loss += *sdiff;
            }
        }

        zero_grad();

        loss->backward();

        if (loss->get_data() > prev)
            lr /= 1.00001;

        for (Value *p : get_params())
            p->moddata(lr); // modify learning rate here

        if ((i + 1) % 1000 == 0)
            cout << i + 1 << ". " << loss->get_data() << " " << lr << endl;

        prev = lr;

        loss->deleteChildren();
        delete loss;
    }
}

float norm(float x)
{
    return x;
}

float denorm(float x)
{
    return x;
}

float f(float x)
{
    // modify this function accordingly
    // note:- if the function is normalized
    // the model may train better
    // so try to normalize the data
    // to [-1, 1] and dont forget to
    // denormalize the output
    return tan(x);
}

int main()
{
    // modify the structure of the net here
    MLP n(1, {8, 8, 1});

    vector<vector<float>> X, Y;

    X = {
        {0.0},
        {0.1},
        {0.2},
        {0.3},
        {0.4},
        {0.5},
        {0.6},
        {0.7},
        {0.8},
        {0.9},
        {1.0},
    };
    Y = {
        {0.000},
        {1.400},
        {1.979},
        {2.425},
        {2.798},
        {3.132},
        {3.431},
        {3.708},
        {3.966},
        {4.208},
        {4.427},
    };

    for (auto &y : Y)
    {
        y[0] = norm(y[0]);
    }

    for (float i = 0; i <= 1; i += 0.1)
    {
        // try to predict different mathematical functions
        X.push_back({i});
        // Y.push_back({sin(i)});
        // Y.push_back({(float)log(1 + pow(i, 2))});
        Y.push_back({f(i)});
    }

    // args:- input, output, epochs
    n.train(X, Y, 100000);

    // predict using the mlp.predict method and print the denormalized output
    auto pred = n.predict({3.1415 / 4});
    cout << "Predicted output:- " << denorm(pred[0]->get_data()) << endl;

    return 0;
}
