#pragma once

#include "value.hpp"

Val operator+(Val a, Val b);
Val operator*(Val a, Val b);
Val operator-(Val a, Val b);
Val operator/(Val a, Val b);

Val exp(Val a);
Val sin(Val a);
Val log(Val a);

Val tanh(Val a);
Val relu(Val a);

Val se(Val a, Val b);
Val ae(Val a, Val b);
Val huber(Val a, Val b);