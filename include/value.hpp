#pragma once 

#include <vector>
#include <unordered_set>
#include <memory>
#include <functional>

class Value;
using Val = std::shared_ptr<Value>;
using WVal = std::weak_ptr<Value>;

inline Val make_val(double data, std::vector<Val> children = {}, char op = 0) {
    return std::make_shared<Value>(data, std::move(children), op);
}

class Value : public std::enable_shared_from_this<Value>
{
    char op;
    std::vector<Val> prev;

    void build_topo(
        Val v, 
        std::vector<Val>& topo, 
        std::unordered_set<Value*>& visited
    );

public:
    double data, grad;
    std::function<void()> _backward;

    Value(
        double _data = 0, 
        std::vector<Val> children = {}, 
        char _op = 0
    );

    void backward();
    void moddata(float lr);
    void zero_grad();
};
