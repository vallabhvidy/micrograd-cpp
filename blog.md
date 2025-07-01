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
float f(float x)
{
    return 2*(float)pow(2, sin(x))-3;
}
```

our MLP will map to this function. 

Note that our MLP does not know anything about this function, we just use it to generate our inputs and outputs(X and Y respectively). 

Input and output data will contain points of this functions. They are generated using the following `for` loop

```
for (float i = 0; i <= 1; i += 0.1)
{
    // try to predict different mathematical functions
    X.push_back({i});
    Y.push_back({f(i)});
}
```

If you want to give your own data(remember our little experiment? :-) ) you can directly feed it to the model like so

```
X = {
    {0.0},
    {0.1},
    {0.2},
    {0.3},
    {0.4},
    {0.5},
    {0.6},
    {0.7},
    {0.8},
    {0.9},
    {1.0},
};

Y = {
    {0.000},
    {1.400},
    {1.979},
    {2.425},
    {2.798},
    {3.132},
    {3.431},
    {3.708},
    {3.966},
    {4.208},
    {4.427},
};
```

It is not necessary but generally recommended to normalize `X` and `Y` so that the model can train better. You can provide your normalization and denormalization logic in these functions,

```
float norm(float x)
{
    return x;
}

float denorm(float x)
{
    return x;
}
```
  
After preparing the training data generate an object of the MLP class. You can pass the structure of the MLP in its constructor.

```
// modify the structure of the net here
MLP n(1, {8, 8, 1});
```

To train the model on our prepared data just call the `train` method. Pass the input, output, and epochs as arguements in the same order. Epochs is just the number of times the model should go through the entire data to modify its weights and biases.

```
// args:- input, output, epochs
n.train(X, Y, 10000);
```

After the model is trained new predictions can be made using the `predict` method. Note that the `predict` method returns an array `Value` objects(in our example it has a single element). `Value` is a custom datatype used to develop the MLP. The actual prediction can be retrieved using the `get_data` method as so, 

```
// predict using the mlp.predict method and print the denormalized output

auto pred = n.predict({3.1415 / 4});
float data = denorm(pred[0]->get_data());

cout << "Predicted output:- " << data << endl;
```

One of the major considerations while training a neural network is the learning rate strategy. Learning rate is a constant which determines the amount of change the weights and biases of the model undergo given there gradients. If the learning rate is high the model may learn faster but it is unstable, whereas a low learning rate makes the model learn very slow. Hence, there are various strategies to determine a suitable learning rate during training. It may be constant during the entire training or it may be determined per epoch.

You can implement a learning rate strategy directly in the `train` method of the `MLP` class.

```
// implement your training strategy here
if (loss->get_data() > prev)
    lr /= 1.00001;
```

The strategy in the example reduces the learning rate by a tiny amount whenever the current loss is greater than the previous loss.