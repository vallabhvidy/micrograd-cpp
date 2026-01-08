#include "../include/value.hpp"
#include "../include/ops.hpp"
#include <vector>
#include <set>
using std::vector, std::unordered_set;

Value::Value(double _data, vector<Val> children, char _op)
{
    data = _data;
    prev = std::move(children);
    op  = _op;
    grad = 0;
    _backward = [] () {};
}

void Value::moddata(float lr)
{
    data += ((-lr) * grad);
}

void Value::backward()
{
    vector<Val> topo;
    unordered_set<Value*> visited;

    build_topo(shared_from_this(), topo, visited);

    grad = 1;

    for (int i = topo.size() - 1; i >= 0; i--)
        topo[i]->_backward();
}

void Value::build_topo(Val v, vector<Val> &topo, unordered_set<Value*> &visited)
{
    // Val v = vweak.lock();

    if (v == nullptr)
        return;

    if (visited.find(v.get()) == visited.end())
    {
        visited.insert(v.get());

        for (Val child : v->prev)
            build_topo(child, topo, visited);

        topo.push_back(v);
    }
}

void Value::zero_grad()
{
    grad = 0;
}
