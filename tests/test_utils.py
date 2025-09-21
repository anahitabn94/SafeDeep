import numpy as np
from utils import get_layer_shapes, str2bool


def test_get_layer_shapes():
    # Mock weights for a small 4-layer network
    W = [np.zeros((3, 2)), np.zeros(2), np.zeros((4, 3)), np.zeros(3), np.zeros((2, 4)), np.zeros(2)]

    # Extract layer information from mock weights
    n_layers, n_neu, n_neu_cum = get_layer_shapes(W)

    assert n_layers == 4
    assert n_neu[0] == [3]
    assert n_neu[1] == [4, 4]
    assert n_neu[3] == [2]
    assert isinstance(n_neu_cum[2], list)


def test_str2bool():
    # Function should correctly map strings to boolean values
    assert str2bool("Yes") is True
    assert str2bool("0") is False
