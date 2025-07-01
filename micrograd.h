#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <vector>
#include <unordered_set>
#include <cstdlib>

class MLP;

class Value
{
    float data;
    std::vector<Value *> prev;
    char op;
    float grad;
    int power;

    void deleteChildren(Value *ptr, std::unordered_set<Value *> &visited);

public:
    bool temp;

    Value(float _data = (std::rand() - (float)(RAND_MAX) / 2) / ((float)(RAND_MAX) * 10), std::vector<Value *> children = {}, char _op = 0, bool _temp = true);
    float get_data();
    float get_grad();
    friend Value *operator+(const Value &a, const Value &b);
    friend Value *operator*(const Value &a, const Value &b);
    friend Value *tanh(const Value &a);
    friend Value *exp(const Value &a);
    friend Value *operator-(const Value &a, const Value &b);
    friend Value *operator/(const Value &a, const Value &b);
    friend Value *operator^(Value &a, const int b);
    friend Value *relu(const Value &a);
    friend Value *sqrdiff(const Value &a, const Value &b);
    friend void operator+=(Value &a, const Value &b);
    void _backward();
    void backward();
    void build_topo(Value *v, std::vector<Value *> &topo, std::unordered_set<Value *> &visited);
    void zero_grad();
    void moddata(float lr);
    void deleteChildren();
};

class Neuron
{
    std::vector<Value *> w;
    Value *b;

public:
    Neuron(int nin);
    Value *predict(std::vector<Value *> in, bool lin = false);
    void zero_grad();
    std::vector<Value *> get_params();
};

class Layer
{
    std::vector<Neuron> neurons;

public:
    Layer(int nin, int nout);
    std::vector<Value *> predict(std::vector<Value *> in, bool lin = false);
    void zero_grad();
    std::vector<Value *> get_params();
};

class MLP
{
    std::vector<Layer> layers;
    float xmin, xmax, ymin, ymax;

    void zero_grad();
    std::vector<Value *> get_params();

public:
    MLP(int nin, std::vector<int> nout);
    std::vector<Value *> predict(std::vector<float> in);
    void train(std::vector<std::vector<float>> xs, std::vector<std::vector<float>> ys, int epoch);
};

#endif