import numpy as np
from utils import get_layer_shapes
from main import analyze_sample


def test_analyze_sample_mock():
    # Tiny mock dataset
    data = np.array([[0.1, 0.2], [0.3, 0.4]])
    lbl = np.array([[0], [1]])

    # Mock prediction (correct)
    pred = np.array([[0.9, 0.1], [0.1, 0.9]])

    # Mock weights for 1 hidden layer
    W = [np.ones((2, 2)), np.zeros(2), np.ones((2, 2)), np.zeros(2)]
    n_layers, n_neu, _ = get_layer_shapes(W)

    # Analyze first sample
    correct, robust = analyze_sample(0, data, lbl, pred, None, W, delta=0.01, flag_lp=False, n_layers=n_layers,
                                     n_neu=n_neu)

    assert correct is True
    assert robust in [True, False]
