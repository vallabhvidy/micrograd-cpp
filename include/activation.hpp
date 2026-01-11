#pragma once
#include "ops.hpp"

enum class Activation {
    ReLU,
    Tanh,
    Lin,
};

inline Val activate(Val x, Activation act) {
    switch (act) {
        case Activation::ReLU: return relu(x);
        case Activation::Tanh: return tanh(x);
        default: return x;
    }
}