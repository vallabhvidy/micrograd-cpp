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
    return log(x);
}

int main()
{
    // modify the structure of the net here
    MLP n(1, {8, 8, 1});

    vector<vector<double>> X, Y;

    // X = {
    //     {0.0},
    //     {0.1},
    //     {0.2},
    //     {0.3},
    //     {0.4},
    //     {0.5},
    //     {0.6},
    //     {0.7},
    //     {0.8},
    //     {0.9},
    //     {1.0},
    // };
    // Y = {
    //     {0.000},
    //     {1.400},
    //     {1.979},
    //     {2.425},
    //     {2.798},
    //     {3.132},
    //     {3.431},
    //     {3.708},
    //     {3.966},
    //     {4.208},
    //     {4.427},
    // };

    for (float i = 1; i <= 3; i += 0.1)
    {
        // try to predict different mathematical functions
        X.push_back({i});
        // Y.push_back({sin(i)});
        // Y.push_back({(float)log(1 + pow(i, 2))});
        Y.push_back({f(i)});
    }

    Normalizer nx, ny;

    auto X_norm = nx.fit(X);
    auto Y_norm = ny.fit(Y);

    // args:- input, output, epochs
    n.train(X_norm, Y_norm, 5000);

    // predict using the mlp.predict method and print the denormalized output
    auto pred = n.predict({nx.norm((float)exp(1))});
    float data = ny.denorm(pred[0]->data);
    cout << "Predicted output:- " << data << endl;

    return 0;
}