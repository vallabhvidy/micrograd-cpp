#include "value.hpp"
#include "ops.hpp"
#include "activation.hpp"
#include "error.hpp"

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
    std::vector<Val> get_params();
};

class Layer
{
    std::vector<Neuron> neurons;
    Activation activation;

public:
    Layer(int nin, int nout, Activation act = Activation::Lin);
    std::vector<Val> predict(std::vector<Val>& in);
};

class MLP
{
    std::vector<Layer> layers;

    std::vector<Val> _predict(std::vector<double> in);
    std::vector<Val> _predict(std::vector<Val>& in);

public:
    MLP(int nin, std::vector<std::pair<int, Activation>> nout);
    std::vector<double> predict(std::vector<double> in);
    std::vector<double> predict(std::vector<Val>& in);
    void train(std::vector<std::vector<Val>>& xs, std::vector<std::vector<Val>>& ys, int epoch, Error error = Error::MSE);
    void train(std::vector<std::vector<double>>& xs, std::vector<std::vector<double>>& ys, int epoch, Error error = Error::MSE);
};

