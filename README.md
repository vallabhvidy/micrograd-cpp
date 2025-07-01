# This is a Neural Network Engine in C++

This is the reimplementation of Andrej Karpathy's Micrograd but in C++.

Test the model by modifing the following code in micrograd.cpp
and run the file to see the model learn!
```
float f(float x)
{
    // modify this function accordingly
    // note:- if the function is normalized
    // the model may train better
    // so try to normalize the data
    // to [-1, 1] and dont forget to
    // denormalize the output
    return tan(x);
}
```

To test, modify the above function then compile and run.
```
$ g++ -o micrograd micrograd.cpp
$ ./micrograd
```
