#ifndef MICROGRAD_H
#define MICROGRAD_H

class Value {
    private:
        float value;
        char op;
        float grad;
        Value* prev1;
        Value* prev2;
        void (Value::*backward) (Value&, Value&);
    public:
        Value(float value, char op = '\0', Value* prev1 = nullptr, Value* prev2 = nullptr);
        Value operator+(Value& other);
        void diff_add(Value& other, Value& out);
        Value operator*(Value& other);
        void diff_mul(Value& other, Value& out);
        float get_value();
};

#endif