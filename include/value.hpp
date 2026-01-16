#pragma once 

#include <vector>
#include <unordered_set>
#include <memory>
#include <functional>

class Value;
using Val = std::shared_ptr<Value>;
using WVal = std::weak_ptr<Value>;

inline Val make_val(double data, std::vector<Val> children = {}, char _op = '\0') {
    return std::make_shared<Value>(data, std::move(children));
}

class Value : public std::enable_shared_from_this<Value>
{
    std::vector<Val> prev;
    std::vector<Val> topo;
    bool param;

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
        char _op = '\0'
    );

    void moddata(float lr);
    void backward(bool force_rebuild = true);
    void zero_grad(bool force_rebuild = true);
    std::vector<Val> get_params(bool force_rebuild = true);
    void set_param() { param = true; }
};
