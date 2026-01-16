#pragma once
#include "ops.hpp"

enum class Error {
    MSE,
    MAE,
    Huber,
    CrossEntropy
};

Val mse(std::vector<Val>& ypred, std::vector<Val>& yact);
Val mae(std::vector<Val>& ypred, std::vector<Val>& yact);
Val huber_loss(std::vector<Val>& ypred, std::vector<Val>& yact);
Val cross_entropy(std::vector<Val>& ypred, std::vector<Val>& yact);

inline Val get_error(std::vector<Val>& ypred, std::vector<Val>& yact, Error act) {
    switch (act) {
        case Error::MSE: return mse(ypred, yact);
        case Error::MAE: return mae(ypred, yact);
        case Error::Huber: return huber_loss(ypred, yact);
        case Error::CrossEntropy: return cross_entropy(ypred, yact);
        default: return mae(ypred, yact);
    }
}