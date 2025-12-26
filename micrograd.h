#ifndef MICROGRAD_H
#define MICROGRAD_H

class Value {
    double value;

public:
    Value(double _value = 0): value(_value) {}
    void iRandom();
    void iZero();
    void iUnity();
};

#endif