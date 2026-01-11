#include "value.hpp"
#include "ops.hpp"
#include "activation.hpp"

class Neuron;
class Layer;
class MLP;

class Neuron
{
    std::vector<Val> w;
    Val b;

public:
    Neuron(int nin);
    Val predict(std::vector<Val>& in);
    void zero_grad();
    std::vector<Val> get_params();
};

class Layer
{
    std::vector<Neuron> neurons;
    Activation activation;

public:
    Layer(int nin, int nout, Activation act = Activation::Lin);
    std::vector<Val> predict(std::vector<Val>& in);
    void zero_grad();
    std::vector<Val> get_params();
};

class MLP
{
    std::vector<Layer> layers;

    void zero_grad();
    std::vector<Val> get_params();

public:
    MLP(int nin, std::vector<std::pair<int, Activation>> nout);
    std::vector<Val> predict(std::vector<double> in);
    void train(std::vector<std::vector<double>> xs, std::vector<std::vector<double>> ys, int epoch);
};

