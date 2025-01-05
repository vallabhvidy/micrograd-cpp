#include "micrograd.h"

Value::Value(float _value, char _op, Value* _prev1, Value* _prev2) {
    value = _value;
    op = _op;
    prev1 = _prev1;
    prev2 = _prev2;
    grad = 0;
    backward = nullptr;
}

Value Value::operator+(Value& other) {
    Value out(value + other.value, '+', this, &other);
    out.backward = Value::diff_add;
    return out;
}

void Value::diff_add(Value& other, Value& out) {
    grad += out.grad;
    other.grad += out.grad;
}

Value Value::operator*(Value& other) {
    Value out(value * other.value, '*', this, &other);
    out.backward = diff_mul;
    return out;
}

void Value::diff_mul(Value& other, Value& out) {
    grad = other.value * out.grad;
    other.grad = value * out.grad;
}

float Value::get_value() {
    return value;
}
