import numpy as np
import scipy.io
import tensorflow as tf
from pytictoc import TicToc
from tqdm import tqdm
from propagation import bound_calc, gurobi_par, compute_relations
from optimization import gb_opt
from utils import parse_args, get_layer_shapes


def check_robustness(W, lower_n, upper_n, pred_label, true_label):
    """Run optimization to check if prediction is provably robust."""
    if pred_label != true_label:
        return False
    flag_rev = (true_label == 1)
    _, low_val = gb_opt(W, lower_n, upper_n, minimize=True, reverse=flag_rev)
    return low_val > 0


def analyze_sample(i, data, lbl, pred, model, W, delta, flag_lp, n_layers, n_neu):
    """Analyze a single sample for correctness and robustness."""
    true_label = int(lbl[i][0])
    pred_label = np.argmax(pred[i])
    if pred_label != true_label:
        return False, False

    # --- initialize bounds ---
    lower_n, upper_n = {}, {}
    lower_n[0] = np.reshape(data[i] - delta, [-1, 1])
    upper_n[0] = np.reshape(data[i] + delta, [-1, 1])

    # --- propagate through hidden layers ---
    for s in range(1, n_layers - 1):
        ll, uu = bound_calc(W[2 * (s - 1)], W[2 * (s - 1) + 1], lower_n[s - 1], upper_n[s - 1])
        lower_n[s], upper_n[s] = ll, uu

        if flag_lp and s > 1:
            ll, uu = gurobi_par(W, s, n_neu[s][0], compute_relations(ll, uu), lower_n, upper_n)
            lower_n[s], upper_n[s] = ll, uu

    # --- propagate to output layer ---
    ll, uu = bound_calc(W[-2], W[-1], lower_n[n_layers - 2], upper_n[n_layers - 2])
    lower_n[n_layers - 1], upper_n[n_layers - 1] = ll, uu

    # --- check robustness ---
    robust = check_robustness(W, lower_n, upper_n, pred_label, true_label)

    return True, robust


def main():
    args = parse_args()
    dataset = scipy.io.loadmat(args.dataset)
    data, lbl = dataset['dataset'], dataset['label']

    model = tf.keras.models.load_model(args.network)
    W = model.get_weights()
    n_layers, n_neu, _ = get_layer_shapes(W)

    # precompute predictions
    pred = model(data, training=False).numpy()

    t_all = TicToc(); t_all.tic()
    cc_data, verified = 0, 0

    for i in tqdm(range(len(lbl)), desc="Analyzing samples"):
        correct, robust = analyze_sample(i, data, lbl, pred, model, W, args.delta, args.lp, n_layers, n_neu)
        cc_data += correct
        verified += robust

    n_samples = len(lbl)
    print(f"Total samples: {n_samples}")
    print(f"Correctly classified samples: {cc_data}")
    print(f"Provably robust: {verified}")
    print("Elapsed time total:", t_all.tocvalue())


if __name__ == "__main__":
    main()
