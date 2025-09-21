import numpy as np
from multiprocessing import Pool, cpu_count
from itertools import repeat
from optimization import gb_opt


def bound_calc(W_lay, W_b, lower_p, upper_p):
    """Propagate bounds through one layer (ReLU)."""
    W_pos, W_neg = np.maximum(W_lay, 0), np.minimum(W_lay, 0)
    ll_pre = W_pos.T @ lower_p[:, -1] + W_neg.T @ upper_p[:, -1] + W_b
    uu_pre = W_pos.T @ upper_p[:, -1] + W_neg.T @ lower_p[:, -1] + W_b
    return np.column_stack([ll_pre, np.maximum(0, ll_pre)]), np.column_stack([uu_pre, np.maximum(0, uu_pre)])


def compute_relations(ll, uu):
    """Compute neuron relations to identify undetermined neurons."""
    return (-1 * np.minimum(ll[:, 0], 0) * np.maximum(uu[:, 0], 0)) / 2


def gurobi_par(W, layer_idx, num_neurons, relation, lower_n, upper_n):
    """Refine bounds of neurons in a layer using Gurobi in parallel."""
    W_tmp = W[:2 * layer_idx]

    with Pool(cpu_count()) as p:
        lower_tuple = p.starmap(
            _gurobi_worker,
            zip(
                range(num_neurons), repeat(relation), repeat(lower_n),
                repeat(upper_n), repeat(W_tmp), repeat(True)
            )
        )
        upper_tuple = p.starmap(
            _gurobi_worker,
            zip(
                range(num_neurons), repeat(relation), repeat(lower_n),
                repeat(upper_n), repeat(W_tmp), repeat(False)
            )
        )

    ll, uu = np.copy(lower_n[layer_idx]), np.copy(upper_n[layer_idx])
    for i in range(num_neurons):
        if lower_tuple[i] is not None:
            ll[i], uu[i] = lower_tuple[i], upper_tuple[i]
    return ll, uu


def _gurobi_worker(neuron, rel, lower_n, upper_n, W_tmp, minimize=True):
    """Worker for parallel Gurobi optimization of a single neuron."""
    if rel[neuron] == 0:
        return None

    W_tmp_n = W_tmp.copy()
    W_tmp_n[-2] = W_tmp_n[-2][:, neuron].reshape(-1, 1)
    W_tmp_n[-1] = np.array([W_tmp_n[-1][neuron]])

    # Call the updated gb_opt with correct keywords
    _, val = gb_opt(W_tmp_n, lower_n, upper_n, minimize=minimize)
    return val, max(0, val)
