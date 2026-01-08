#pragma once 
#include <vector>
#include <algorithm>
#include <limits>

class Normalizer {
    double mn, mx;

public: 
    Normalizer() {
        mn = std::numeric_limits<double>::max();
        mx = std::numeric_limits<double>::lowest();
    }

    std::vector<std::vector<double>> fit(std::vector<std::vector<double>> a) {
        
        for (auto i: a) {
            for (auto j: i) {
                mn = std::min(mn, j);
                mx = std::max(mx, j);
            }
        }

        for (auto& i: a) {
            for (auto& j: i) {
                j = norm(j);
            }
        }

        return a;
    }

    double norm(double x) {
        double xn = (2 * (x - mn)) / (mx - mn) - 1;
        return xn;
    }

    double denorm(double x) {
        double xd = ((x + 1) * (mx - mn)) / 2 + mn;
        return xd;
    }
};