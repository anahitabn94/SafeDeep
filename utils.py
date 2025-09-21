import os
import argparse
import numpy as np


def get_layer_shapes(W):
    """Return network layer structure."""
    n_layers = len(W) // 2 + 1
    n_neu, n_neu_cum = {}, {}
    
    for k in range(n_layers):
        if k == 0:  # input layer
            n_neu[k] = [W[0].shape[0]]
            n_neu_cum[k] = [W[0].shape[0]]
        elif k == n_layers - 1:  # output layer
            n_neu[k] = [W[-1].shape[0]]
            n_neu_cum[k] = [n_neu_cum[k - 1][-1] + np.sum(n_neu[k])]
        else:  # hidden layer
            size = W[2 * k].shape[0]
            n_neu[k] = [size, size]
            n_neu_cum[k] = [
                n_neu_cum[k - 1][-1] + size,
                n_neu_cum[k - 1][-1] + np.sum(n_neu[k])
            ]
    
    return n_layers, n_neu, n_neu_cum


def check_file_extension(fname, valid_exts):
    _, ext = os.path.splitext(fname)
    if ext not in valid_exts:
        raise argparse.ArgumentTypeError(f"only {', '.join(valid_exts)} formats supported")
    return fname


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="SafeDeep")
    parser.add_argument("--network", type=lambda f: check_file_extension(f, [".h5"]),
                        default="./models/patient_01.h5", help="Neural network file")
    parser.add_argument("--dataset", type=lambda f: check_file_extension(f, [".mat"]),
                        default="./datasets/patient_01.mat", help="Dataset file")
    parser.add_argument("--delta", type=float, default=0.005, help="Perturbation")
    parser.add_argument("--lp", type=str2bool, default=True, help="Whether to use LP or not")
    return parser.parse_args()
