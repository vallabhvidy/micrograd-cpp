# micrograd-cpp

'[micrograd-cpp](https://github.com/vallabhvidy/micrograd-cpp)' is a reimplementation of Andrej Karpathy's '[micrograd](https://github.com/karpathy/micrograd)'. 

> 'micrograd' is a tiny Autograd engine (with a bite! :)). Implements backpropagation (reverse-mode autodiff) over a dynamically built DAG and a small neural networks library on top of it with a PyTorch-like API. Both are tiny, with about 100 and 50 lines of code respectively. The DAG only operates over scalar values, so e.g. we chop up each neuron into all of its individual tiny adds and multiplies. However, this is enough to build up entire deep neural nets doing binary classification, as the demo notebook shows. Potentially useful for educational purposes.

micrograd-cpp is just micrograd but written in C++. It was created to test both C++ proficiency and understanding of concepts like neural networks, backpropagation, etc.

Contents of this article:-  
1. What is micrograd?  
2. How to use micrograd-cpp?  
3. Where do micrograd and micrograd-cpp differ?  


## What is micrograd?

*micrograd* is basically a black box that takes a set of inputs and outputs and configures itself in such a way that if you give it new inputs it can give an approximation of the corresponding outputs.

But wait! what problem does this thing actaully solve? well lets consider that you were born before Newton and dont know how gravity works. You perform an experiment where you drop an object from different heights and record the speed at which they hit the ground. You just want to know at what speed an object will fall on the ground if dropped from a certain height from which you haven't done the experiment already. 

One way to achieve this to figure out the formula of gravity like Newton did buts lets say you are lazy and dont really need or want to devise the exact formula and an approximation is also fine.

Now micrograd comes into picture. You just need to give the heights and the corresponding speeds to it and it will adjust its internals in such a way that if you give it a new height it will give you a nice approximation of the speed that the object will have if droppend from this height.

How does this black box work? well maybe i will explain it in a seperate article...


## How to use micrograd-cpp?

if you look at `micrograd.cpp` you will find a function, 

```
float f(float x) {
    // modify this function accordingly
    // and make sure it normalized to [-1, 1]
    return 2*(float)pow(2, sin(x))-3;
}
```

our MLP will map to this function. 

Note that our MLP does not know anything about this function, we just use it to generate our inputs and outputs. 

