#include "../include/error.hpp"

Val mse(std::vector<Val>& ypred, std::vector<Val>& yact) {
    Val loss = make_val(0.00);
    int n = ypred.size();
    for (int i = 0; i < n; i++) {
        loss = loss + se(ypred[i], yact[i]);
    }

    Val inv_n = make_val(1.0 / n);
    loss = loss * inv_n;
    return loss;
}

Val mae(std::vector<Val>& ypred, std::vector<Val>& yact) {
    Val loss = make_val(0.00);
    int n = ypred.size();
    for (int i = 0; i < n; i++) {
        loss = loss + ae(ypred[i], yact[i]);
    }

    Val inv_n = make_val(1.0 / n);
    loss = loss * inv_n;
    return loss;
}

Val huber_loss(std::vector<Val>& ypred, std::vector<Val>& yact) {
    Val loss = make_val(0.00);
    int n = ypred.size();
    for (int i = 0; i < n; i++) {
        loss = loss + huber(ypred[i], yact[i]);
    }

    Val inv_n = make_val(1.0 / n);
    loss = loss * inv_n;

    return loss;
}

Val cross_entropy(std::vector<Val>& ypred, std::vector<Val>& yact) {
    std::vector<Val>& logits = ypred;
    Val max_logit = make_val(0.0);
    for (Val& logit: logits) {
        if (logit->data > max_logit->data)
            max_logit = logit;
    }

    Val sum_counts = make_val(0.0);
    for (Val& logit: logits) {
        sum_counts = sum_counts + exp(logit - max_logit);
    }

    Val log_sum_counts = log(sum_counts) + max_logit;
    int n = ypred.size();

    Val loss = make_val(0.0);
    for (int i = 0; i < n; i++) {
        loss = loss - yact[i] * (ypred[i] - log_sum_counts);
    }

    return loss;
}