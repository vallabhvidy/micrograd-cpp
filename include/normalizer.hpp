#pragma once 
#include <vector>
#include <limits>

class Normalizer {
    double mn, mx;

public: 
    Normalizer():
        mn(std::numeric_limits<double>::max()),
        mx(std::numeric_limits<double>::lowest()) {}

    std::vector<std::vector<double>> fit(std::vector<std::vector<double>> a);

    double norm(double x);
    double denorm(double x);
};