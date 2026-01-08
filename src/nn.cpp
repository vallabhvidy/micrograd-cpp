#include "../include/nn.hpp"
#include <vector>
#include <iostream>
#include <random>
using std::vector, std::cout, std::endl;

Neuron::Neuron(int nin)
{
    static std::mt19937 gen(24);
    std::normal_distribution<double> dist(
        0.0f,
        1.0f / std::sqrt(nin)
    );
    w = vector<Val>(nin);
    b = std::make_shared<Value>(Value(dist(gen)));
    for (auto &weight: w)
    {
        weight = std::make_shared<Value>(Value(dist(gen)));
    }
}

Layer::Layer(int nin, int nout)
{
    neurons.clear();
    for (int i = 0; i < nout; i++)
        neurons.emplace_back(Neuron(nin));
}

MLP::MLP(int nin, vector<int> nout)
{
    layers.push_back(Layer(nin, nout[0]));
    for (int i = 0; i < (int)nout.size() - 1; i++)
        layers.push_back(Layer(nout[i], nout[i + 1]));
}

vector<Val> Neuron::get_params()
{
    vector<Val> params = w;
    params.push_back(b);
    return params;
}

vector<Val> Layer::get_params()
{
    vector<Val> params;
    for (Neuron &n : neurons)
    {
        vector<Val > n_params = n.get_params();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}

vector<Val> MLP::get_params()
{
    vector<Val > params;
    for (Layer &l : layers)
    {
        vector<Val> l_params = l.get_params();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}

void Neuron::zero_grad()
{
    for (Val weight : w)
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

Val Neuron::predict(vector<Val> in, bool lin)
{
    if (in.size() != w.size())
    {
        cout << "Input size does not match" << endl;
        return nullptr;
    }
    Val sum = std::make_shared<Value>(Value(0));
    sum = sum + b;
    for (int i = 0; i < (int)w.size(); i++)
    {
        Val val = in[i];
        Val mul = w[i] * val;
        sum = sum + mul;
    }

    if (!lin)
        sum = sin(sum);

    return sum;
}

vector<Val> Layer::predict(vector<Val > in, bool lin)
{
    vector<Val > act;
    for (Neuron& n : neurons)
        act.push_back(n.predict(in, lin));

    return act;
}

vector<Val> MLP::predict(vector<double> in)
{
    vector<Val> x;
    for (double i : in)
        x.push_back(std::make_shared<Value>(Value(i)));
    for (int i = 0; i < (int)layers.size() - 1; i++)
        x = layers[i].predict(x);

    x = layers.back().predict(x, true);

    return x;
}

void MLP::train(vector<vector<double>> xs, vector<vector<double>> ys, int epoch)
{
    cout << "Loss:- " << endl;
    double lr = 0.023;
    for (int i = 0; i < epoch; i++)
    {
        Val loss = std::make_shared<Value>(Value(0.00));
        
        for (int j = 0; j < (int)ys.size(); j++)
        {
            vector<Val> ypred = predict(xs[j]);
            for (int k = 0; k < (int)ypred.size(); k++)
            {
                Val y = std::make_shared<Value>(Value(ys[j][k]));
                Val sdiff = sqrdiff(ypred[k], y);
                loss = loss + sdiff;
            }
        }

        zero_grad();

        loss->backward();

        for (Val p : get_params())
            p->moddata(lr);

        if ((i + 1) % 1000 == 0)
            cout << i + 1 << ". " << loss->data << " " << lr << endl;

    }
}
