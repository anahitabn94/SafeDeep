import numpy as np
from propagation import bound_calc, compute_relations


def test_bound_calc_shapes():
    # Mock weight matrix and bias vector for one layer
    W_lay = np.array([[1.0, -1.0], [0.5, 0.5]])
    W_b = np.array([0.1, -0.1])

    # Mock lower/upper neuron bounds for inputs
    lower = np.array([[0.0], [1.0]])
    upper = np.array([[1.0], [2.0]])

    # Call bound calculation function
    ll, uu = bound_calc(W_lay, W_b, lower, upper)

    assert ll.shape == (2, 2)
    assert uu.shape == (2, 2)


def test_compute_relations_values():
    # Lower and upper bounds for two neurons
    ll = np.array([[0.0], [-1.0]])
    uu = np.array([[1.0], [2.0]])

    # Compute ReLU relations
    rel = compute_relations(ll, uu)

    assert rel.shape[0] == ll.shape[0]
