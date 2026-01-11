#include "../include/nn.hpp"
#include <vector>
#include <iostream>
#include <random>
using std::vector, std::cout, std::endl, std::pair;

Neuron::Neuron(int nin)
{
    static std::mt19937 gen(24);
    std::normal_distribution<double> dist(
        0.0f,
        1.0f / std::sqrt(nin)
    );
    w = vector<Val>(nin);
    b = make_val(dist(gen));
    for (Val &weight: w)
    {
        weight = make_val(dist(gen));
    }
}

Layer::Layer(int nin, int nout, Activation _activation)
{
    activation = _activation; 
    neurons.clear();
    for (int i = 0; i < nout; i++)
        neurons.emplace_back(Neuron(nin));
}

MLP::MLP(int nin, vector<pair<int, Activation>> nout)
{
    layers.push_back(Layer(nin, nout[0].first, nout[0].second));
    for (int i = 0; i < (int)nout.size() - 1; i++)
        layers.push_back(Layer(nout[i].first, nout[i + 1].first, nout[i+1].second));
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
        vector<Val> n_params = n.get_params();
        params.insert(params.end(), n_params.begin(), n_params.end());
    }
    return params;
}

vector<Val> MLP::get_params()
{
    vector<Val> params;
    for (Layer &l : layers)
    {
        vector<Val> l_params = l.get_params();
        params.insert(params.end(), l_params.begin(), l_params.end());
    }
    return params;
}

void Neuron::zero_grad()
{
    for (Val &weight : w)
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

Val Neuron::predict(vector<Val>& in)
{
    if (in.size() != w.size())
    {
        cout << "Input size does not match" << endl;
        return nullptr;
    }
    Val sum = make_val(0);
    sum = sum + b;
    for (int i = 0; i < (int)w.size(); i++)
    {
        Val val = in[i];
        Val mul = w[i] * val;
        sum = sum + mul;
    }

    return sum;
}

vector<Val> Layer::predict(vector<Val>& in)
{
    vector<Val> act;
    for (Neuron &n : neurons)
        act.push_back(activate(n.predict(in), activation));

    return act;
}

vector<Val> MLP::predict(vector<double> in)
{
    vector<Val> x;
    for (double i : in)
        x.push_back(make_val(i));
    for (int i = 0; i < (int)layers.size(); i++)
        x = layers[i].predict(x);

    return x;
}

void MLP::train(vector<vector<double>> xs, vector<vector<double>> ys, int epoch)
{
    cout << "Training... " << endl;
    double lr = 0.002;
    int n = ys.size();
    for (int i = 0; i < epoch; i++)
    {
        Val loss = make_val(0.00);
        
        for (int j = 0; j < (int)ys.size(); j++)
        {
            vector<Val> ypred = predict(xs[j]);
            for (int k = 0; k < (int)ypred.size(); k++)
            {
                Val y = make_val(ys[j][k]);
                Val sdiff = sqrdiff(ypred[k], y);
                loss = loss + sdiff;
            }
        }

        loss->zero_grad();

        loss->backward();

        for (Val p : get_params())
            p->moddata(lr);

        if ((i + 1) % 100 == 0 || i == 0)
        {
            double avg_loss = loss->data / n;

            double grad_norm = 0.0;
            for (Val p : get_params())
                grad_norm += p->grad * p->grad;
            grad_norm = std::sqrt(grad_norm);

            cout << "[Epoch " << i + 1 << "/" << epoch << "] "
                 << "loss=" << avg_loss
                 << " grad_norm=" << grad_norm
                 << " lr=" << lr << endl;
        }

    }
}
