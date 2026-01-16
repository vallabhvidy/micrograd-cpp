#include "include/nn.hpp"
#include <fstream>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>
using namespace std;

vector<double> one_hot(int c, int num) {
    vector<double> enc;
    for (int i = 0; i < num; i++) {
        if (i == c) enc.push_back(1);
        else enc.push_back(0);
    }

    return enc;
}

vector<Val> one_hot_val(int c, int num) {
    vector<Val> enc;
    for (int i = 0; i < num; i++) {
        if (i == c) enc.push_back(make_val(1.0));
        else enc.push_back(make_val(0.0));
    }

    return enc;
}

vector<double> softmax(vector<double> x) {
    double total = 0;
    for (double& xi: x) total += exp(xi);

    for (double& xi: x) xi /= total;

    return x;
}

int sample(vector<double> pred) {
    static random_device rd;
    static mt19937 gen(rd());
    
    vector<double> w;
    double mx = *max_element(pred.begin(), pred.end());
    for (double p: pred) w.push_back(exp(p - mx));

    discrete_distribution<int> d(w.begin(), w.end());

    return d(gen);
}

int main() {
    ifstream f("names.txt");
    vector<string> names;
    string name;
    while (getline(f, name)) {
        names.push_back(name);
    }

    cout << "Number of names : " << names.size() << endl;

    set<char> vocab;
    for (string name: names) {
        for (char c: name) {
            vocab.insert(c);
        }
    }
    vocab.insert('.');
    cout << "Vocab size : " << vocab.size() << endl;
    vector<char> itoc(vocab.begin(), vocab.end());
    sort(itoc.begin(), itoc.end());

    map<char, int> ctoi;
    for (int i = 0; i < (int)itoc.size(); i++) {
        ctoi[itoc[i]] = i;
    }

    int block_size = 8;
    cout << "Block size : " << block_size << endl;
    vector<vector<Val>> X, Y;

    int emb_size = 10;
    vector<vector<Val>> emb(vocab.size(), vector<Val>(emb_size));
    static std::mt19937 gen(24);
    std::normal_distribution<double> dist(0.0f, 1.0f);
    for (vector<Val>& i: emb) {
        for (Val& j: i) {
            j = make_val(dist(gen));
            j->set_param();
        }
    }
    
    for (string name: names) {
        string context(block_size, '.');
        name.push_back('.');
        for (int i = 0; i < (int)name.size(); i++) {
            X.push_back({});
            for (int j = 0; j < (int)context.size(); j++) {
                vector<Val>& cur = emb[ctoi[context[j]]];
                X.back().insert(X.back().end(), cur.begin(), cur.end());
            }
            Y.push_back(one_hot_val(ctoi[name[i]], vocab.size()));
            context = context.substr(1) + name[i];
        }
    }

    cout << "Input shape : (" << X.size() << ", " << X[0].size() << ")\n";
    cout << "Output shape : (" << Y.size() << ", " << Y[0].size() << ")\n";

    MLP n(block_size * emb_size, {
        {32, Activation::Tanh},
        {64, Activation::Tanh},
        {vocab.size(), Activation::Lin}
    });

    n.train(X, Y, 10000, Error::CrossEntropy);

    cout << "Sampled names : " << endl;
    for (int i = 0; i < 10; i++) {
        string context(block_size, '.');
        char ch = '\0';
        while (ch != '.') {
            vector<Val> enc;
            for (char c: context) {
                vector<Val>& cur = emb[ctoi[c]];
                enc.insert(enc.end(), cur.begin(), cur.end());
            }

            vector<double> pred = n.predict(enc);
            ch = itoc[sample(pred)];

            cout << ch;
        }
        cout << endl;
    }
}