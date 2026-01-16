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
    b->set_param();
    for (Val &weight: w)
    {
        weight = make_val(dist(gen));
        weight->set_param();
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
        layers.push_back(Layer(nout[i].first, nout[i+1].first, nout[i+1].second));
}

// vector<Val> Neuron::get_params()
// {
//     vector<Val> params = w;
//     params.push_back(b);
//     return params;
// }

// vector<Val> Layer::get_params()
// {
//     vector<Val> params;
//     for (Neuron &n : neurons)
//     {
//         vector<Val> n_params = n.get_params();
//         params.insert(params.end(), n_params.begin(), n_params.end());
//     }
//     return params;
// }

// vector<Val> MLP::get_params()
// {
//     vector<Val> params;
//     for (Layer &l : layers)
//     {
//         vector<Val> l_params = l.get_params();
//         params.insert(params.end(), l_params.begin(), l_params.end());
//     }
//     return params;
// }

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

vector<Val> MLP::_predict(vector<double> in)
{
    vector<Val> x;
    for (double i : in)
        x.push_back(make_val(i));
    for (int i = 0; i < (int)layers.size(); i++)
        x = layers[i].predict(x);

    return x;
}

vector<Val> MLP::_predict(vector<Val>& in)
{
    vector<Val> x = in;
    for (int i = 0; i < (int)layers.size(); i++)
        x = layers[i].predict(x);

    return x;
}

vector<double> MLP::predict(vector<double> in)
{
    vector<Val> x = _predict(in);
    vector<double> out;
    for (Val xi: x) out.push_back(xi->data);
    return out;
}

vector<double> MLP::predict(vector<Val>& in)
{
    vector<Val> x = _predict(in);
    vector<double> out;
    for (Val xi: x) out.push_back(xi->data);
    return out;
}

inline int min(int& a, int& b) {
    return (a > b ? b : a);
}

void MLP::train(vector<vector<Val>>& xs, vector<vector<Val>>& ys, int epochs, Error error)
{
    double lr = 0.01;
    int n = ys.size();
    int batch_size = 2;
    static std::random_device rd;
    static std::mt19937 gen(rd());

    std::uniform_int_distribution<int> dist(0, n-1);

    cout << "Training... " << endl;
    for (int epoch = 1; epoch <= epochs; epoch++)
    {
        Val loss = make_val(0.0);
        for (int j = 0; j < batch_size; j++)
        {
            int i = dist(gen);
            vector<Val> ypred = _predict(xs[i]);
            vector<Val> yact;
            for (Val& y: ys[i]) yact.push_back(y);

            loss = loss + get_error(ypred, yact, error);
        }

        loss = loss * make_val(1.0 / batch_size);

        loss->zero_grad(true);
        loss->backward(false);
        vector<Val> params = loss->get_params(false);

        for (Val p : params)
            p->moddata(lr);

        if (epoch % 100 == 0 || epoch == 1)
        {
            double avg_loss = loss->data;

            double grad_norm = 0.0;
            for (Val p : params)
                grad_norm += p->grad * p->grad;
            grad_norm = std::sqrt(grad_norm);

            cout << "[Epoch " << epoch << "/" << epochs << "]"
                << " loss=" << avg_loss
                << " grad_norm=" << grad_norm
                << " lr=" << lr << endl;
        }

        lr *= 0.99999;
    }
}

void MLP::train(vector<vector<double>>& xs, vector<vector<double>>& ys, int epochs, Error error)
{
    vector<vector<Val>> x_val, y_val;
    for (vector<double>& x: xs) {
        x_val.push_back({});
        for (double& xi: x) {
            x_val.back().push_back(make_val(xi));
        } 
    }
    for (vector<double>& y: ys) {
        y_val.push_back({});
        for (double& yi: y) {
            y_val.back().push_back(make_val(yi));
        } 
    }
    train(x_val, y_val, epochs, error);
}
