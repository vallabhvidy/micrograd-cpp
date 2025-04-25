#include "micrograd.h"
#include <bits/stdc++.h>
using namespace std;
int main(int argc, char* argv[]) {
    Value a(3.0), b(4.0);
    Value c = a + b;
    cout << c.get_data() << endl;
    return 0;
}