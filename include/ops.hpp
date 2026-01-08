#pragma once

#include "value.hpp"

Val operator+(Val a, Val b);
Val operator*(Val a, Val b);
Val operator-(Val a, Val b);
Val operator/(Val a, Val b);

Val tanh(Val a);
Val exp(Val a);

Val relu(Val a);
Val sqrdiff(Val a, Val b);
Val sin(Val a);