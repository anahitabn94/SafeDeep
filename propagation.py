# This function is used to propagate through the neural network to calculate the loose bounds of each neuron.

import numpy as np


def bound_calc(W_lay, W_b, lower_p, upper_p):
    n_neu = np.shape(W_lay)[1]
    num_cols = 2
    ll = np.zeros([n_neu, num_cols])
    uu = np.zeros([n_neu, num_cols])
    for m in range(n_neu):
        W_pos = np.maximum(W_lay[:, m], 0)
        W_neg = np.minimum(W_lay[:, m], 0)
        ll[m, 0] = np.sum(W_pos * lower_p[:, -1] + W_neg * upper_p[:, -1]) + W_b[m]  # lower bound before ReLU
        uu[m, 0] = np.sum(W_pos * upper_p[:, -1] + W_neg * lower_p[:, -1]) + W_b[m]  # upper bound before ReLU
        ll[m, 1] = max(0, ll[m, 0])  # lower bound after ReLU
        uu[m, 1] = max(0, uu[m, 0])  # upper bound after ReLU
    return ll, uu

