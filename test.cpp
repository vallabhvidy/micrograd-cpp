#include "include/nn.hpp"
#include "include/normalizer.hpp"
#include <iostream>
#include <cmath>
using namespace std;

float f(float x)
{
    // modify this function accordingly
    // note:- if the function is normalized
    // the model may train better
    // so try to normalize the data
    // to [-1, 1] and dont forget to
    // denormalize the output
    return sin(x);
}

int main()
{
    // modify the structure of the net here
    MLP n(1, {
        {8, Activation::Tanh},
        {1, Activation::Lin}
    });

    vector<vector<double>> X, Y;

    for (float i = -3; i <= 3; i += 0.2)
    {
        // try to predict different mathematical functions
        X.push_back({i});
        // Y.push_back({sin(i)});
        // Y.push_back({(float)log(1 + pow(i, 2))});
        Y.push_back({f(i)});
    }

    // X = {
    //     {-2.0},
    //     {-1.5},
    //     {-1.0},
    //     {-0.5},
    //     {0.0},
    //     {0.5},
    //     {1.0},
    //     {1.5},
    //     {2.0}
    // };

    // Y = {
    //     {4.0},
    //     {2.25},
    //     {1.0},
    //     {0.25},
    //     {0.0},
    //     {0.25},
    //     {1.0},
    //     {2.25},
    //     {4.0}
    // };

    // Normalizer nx, ny;
    // // uses min-max normalization
    // auto X_norm = nx.fit(X);
    // auto Y_norm = ny.fit(Y);

    // args:- input, output, epochs
    n.train(X, Y, 5000, Error::Huber);

    vector<vector<double>> tests = {
        {-3.14}, {-3.14/2}, {-3.14/3}, {-3.14/6}, {0},
        {3.14}, {3.14/2}, {3.14/3}, {3.14/6}
    };

    cout << endl;

    for (auto test: tests) {
        // double pred = ny.denorm(n.predict({nx.norm(test[0])})[0]->data);
        double pred = n.predict(test)[0];
        cout << "sin(" << test[0] << ")" << " = " << pred << endl;
    }

    return 0;
}