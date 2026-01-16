#include "../include/ops.hpp"
#include <cmath>
#include <iostream>

Val operator+(Val a, Val b)
{   
    Val c = make_val(a->data + b->data, {a, b}, '+');
    WVal wa = a, wb = b, wc = c;
    std::function<void()> _backward = [wa, wb, wc] () {
        Val a = wa.lock(), b = wb.lock(), c = wc.lock();
        if (!a || !b || !c) return;
        a->grad += c->grad;
        b->grad += c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val operator*(Val a, Val b)
{
    Val c = make_val(a->data * b->data, {a, b}, '*');
    WVal wa = a, wb = b, wc = c;
    std::function<void()> _backward = [wa, wb, wc] () {
        Val a = wa.lock(), b = wb.lock(), c = wc.lock();
        if (!a || !b || !c) return;
        a->grad += b->data * c->grad;
        b->grad += a->data * c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val operator-(Val a, Val b)
{
    Val c = make_val(a->data - b->data, {a, b}, '-');
    WVal wa = a, wb = b, wc = c;
    std::function<void()> _backward = [wa, wb, wc] () {
        Val a = wa.lock(), b = wb.lock(), c = wc.lock();
        if (!a || !b || !c) return;
        a->grad += c->grad;
        b->grad -= c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val operator/(Val a, Val b)
{
    if (b->data == 0) {
        std::cerr << "Divide by zero error!" << std::endl;
        exit(1);
    }
    Val c = make_val(a->data / b->data, {a, b}, '/');
    WVal wa = a, wb = b, wc = c;
    std::function<void()> _backward = [wa, wb, wc] () {
        Val a = wa.lock(), b = wb.lock(), c = wc.lock();
        if (!a || !b || !c) return;
        a->grad += c->grad / b->data;
        b->grad += (-c->grad * a->data) / (b->data * b->data);
    };
    c->_backward = _backward;
    return c;
}

Val relu(Val a)
{
    Val c = make_val((a->data > 0 ? a->data : 0.01 * a->data), {a}, 'r');
    WVal wa = a, wc = c;
    std::function<void()> _backward = [wa, wc] () {
        Val a = wa.lock(), c = wc.lock();
        if (!a || !c) return;
        a->grad += (c->data > 0 ? 1 : 0.01) * c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val tanh(Val a)
{
    Val c = make_val(std::tanh(a->data), {a}, 't');
    WVal wa = a, wc = c;
    std::function<void()> _backward = [wa, wc] () {
        Val a = wa.lock(), c = wc.lock();
        if (!a || !c) return;
        a->grad += (1 - c->data * c->data) * c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val exp(Val a)
{
    Val c = make_val(std::exp(a->data), {a}, 'e');
    WVal wa = a, wc = c;
    std::function<void()> _backward = [wa, wc] () {
        Val a = wa.lock(), c = wc.lock();
        if (!a || !c) return;
        a->grad += c->data * c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val log(Val a)
{
    Val c = make_val(std::log(a->data), {a}, 'e');
    WVal wa = a, wc = c;
    std::function<void()> _backward = [wa, wc] () {
        Val a = wa.lock(), c = wc.lock();
        if (!a || !c) return;
        a->grad += c->grad / a->data;
    };
    c->_backward = _backward;
    return c;
}

Val sin(Val a)
{
    Val c = make_val(sin(a->data), {a}, 's');
    WVal wa = a, wc = c;
    std::function<void()> _backward = [wa, wc] () {
        Val a = wa.lock(), c = wc.lock();
        if (!a || !c) return;
        a->grad += std::cos(a->data) * c->grad;
    };
    c->_backward = _backward;
    return c;
}

Val se(Val a, Val b)
{
    double d = a->data - b->data;
    Val c = make_val(d * d, {a, b}, 'q');
    WVal wa = a, wb = b, wc = c;
    std::function<void()> _backward = [wa, wb, wc] () {
        Val a = wa.lock(), b = wb.lock(), c = wc.lock();
        if (!a || !b || !c) return;
        a->grad += 2 * (a->data - b->data) * c->grad;
        b->grad -= 2 * (a->data - b->data) * c->grad; 
    };
    c->_backward = _backward;
    return c;
}

Val ae(Val a, Val b) {
    double d = std::abs(a->data - b->data);
    Val c = make_val(d, {a, b}, 'm');
    WVal wa = a, wb = b, wc = c;
    if (a->data > b->data) {
        c->_backward = [wa, wb, wc] () {
            Val a = wa.lock(), b = wb.lock(), c = wc.lock();
            if (!a || !b || !c) return;
            a->grad += c->grad;
            b->grad -= c->grad; 
        };
    } else {
        c->_backward = [wa, wb, wc] () {
            Val a = wa.lock(), b = wb.lock(), c = wc.lock();
            if (!a || !b || !c) return;
            a->grad -= c->grad;
            b->grad += c->grad; 
        };
    }

    return c;
}

Val huber(Val a, Val b) {
    double d = a->data - b->data;
    double t = 1;
    if (std::abs(d) <= t) {
        double v = (d * d) / 2;
        Val c = make_val(v, {a, b}, 'h');
        WVal wa = a, wb = b, wc = c;
        c->_backward = [wa, wb, wc] () {
            Val a = wa.lock(), b = wb.lock(), c = wc.lock();
            if (!a || !b || !c) return;
            double d = a->data - b->data;
            a->grad += d * c->grad;
            b->grad -= d * c->grad;
        };
        return c;
    } else {
        double v = t * (std::abs(d) - t/2);
        Val c = make_val(v, {a, b}, 'h');
        WVal wa = a, wb = b, wc = c;
        c->_backward = [wa, wb, wc, t] () {
            Val a = wa.lock(), b = wb.lock(), c = wc.lock();
            if (!a || !b || !c) return;
            double d = a->data - b->data;
            int s = (d > 0 ? 1 : -1);

            a->grad += t * s * c->grad;
            b->grad -= t * s * c->grad;
        };
        return c;
    }
}