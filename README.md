# This is a Neural Network Engine in C++

This is the reimplementation of Andrej Karpathy's Micrograd but in C++.

Test the model by modifing the following code in micrograd.cpp
and run the file to see the model learn!
```
float f(float x) {
    // modify this function accordingly
    // and make sure it is normalized to [-1, 1]
    return 2*(float)pow(2, sin(x))-3;
}
```

To test, modify the above function then compile and run.
```
$ g++ -o micrograd micrograd.cpp
$ ./micrograd
```
