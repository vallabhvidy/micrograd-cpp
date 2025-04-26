#ifndef MICROGRAD_H
#define MICROGRAD_H

#include <vector>
#include <set>
#include <cstdlib>

class MLP;

class Value {
    float data;
    std::vector<Value*> prev;
    char op;
    float grad;
    int power;
public: 
    Value(float _data = std::rand() / (float)(RAND_MAX), std::vector<Value*> children = {}, char _op = 0);
    float get_data();
    friend Value operator+(Value& a, Value& b);
    friend Value operator*(Value& a, Value& b);
    friend Value tanh(Value& a);
    friend Value exp(Value& a);
    friend Value operator-(Value& a, Value& b);
    friend Value operator/(Value& a, Value& b);
    friend Value operator^(Value& a, int b);
    friend Value relu(Value& a);
    void _backward();
    void backward();
    void build_topo(Value* v, std::vector<Value*>& topo, std::set<Value*>& visited);
    void zero_grad();
    void moddata(float lr);
};

class Neuron {
    std::vector<Value*> w;
    Value* b = new Value();
public: 
    Neuron(int nin); 
    Value* predict(std::vector<Value*> in, bool lin = false);
    void zero_grad();
    std::vector<Value*> get_params();
};

class Layer {
    std::vector<Neuron> neurons;
public:
    Layer(int nin, int nout);
    std::vector<Value*> predict(std::vector<Value*> in, bool lin = false);
    void zero_grad();
    std::vector<Value*> get_params();
};

class MLP {
    std::vector<Layer> layers;
public:
    MLP(int nin, std::vector<int> nout);
    std::vector<Value*> predict(std::vector<float> in, bool lin = false);
    void zero_grad();
    std::vector<Value*> get_params();
    void train(std::vector<std::vector<float>> xs, std::vector<std::vector<float>> ys, int epoch);
};

#endif