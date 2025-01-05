#include <bits/stdc++.h>
#include "micrograd.h"
using namespace std;
int main(int argc, char* argv[]) {
    Value a(3.0), b(4.0);
    Value c = a + b;
    cout << c.get_value() << endl;
    return 0;
}