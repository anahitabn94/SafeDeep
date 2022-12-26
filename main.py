import numpy as np
import scipy
from pytictoc import TicToc
import tensorflow as tf
from optimization import gb_opt
from propagation import bound_calc
from multiprocessing import Pool
import multiprocessing
from itertools import repeat
import argparse
import os


def gurobi_par_l(neuron, rel, lower_n, upper_n, W_tmp):  # calculate the lower bound of neurons in parallel
    if rel[neuron] != 0:  # check whether the neuron has over-approximation
        W_tmp_n = W_tmp.copy()
        W_tmp_n[-2] = np.reshape(W_tmp_n[-2][:, neuron], [-1, 1])  # weight
        W_tmp_n[-1] = np.reshape(W_tmp_n[-1][neuron], [1, ])  # bias
        _, low = gb_opt(W_tmp_n, lower_n, upper_n, flag_min=True)  # optimization function using Gurobi
        return low, max(0, low)


def gurobi_par_u(neuron, rel, lower_n, upper_n, W_tmp):  # calculate the upper bound of neurons in parallel
    if rel[neuron] != 0:  # check whether the neuron has over-approximation
        W_tmp_n = W_tmp.copy()
        W_tmp_n[-2] = np.reshape(W_tmp_n[-2][:, neuron], [-1, 1])  # weight
        W_tmp_n[-1] = np.reshape(W_tmp_n[-1][neuron], [1, ])  # bias
        _, up = gb_opt(W_tmp_n, lower_n, upper_n, flag_min=False)  # optimization function using Gurobi
        return up, max(0, up)


def is_network(fname):  # check whether the neural network format is supported by the framework.
    _, ext = os.path.splitext(fname)
    if ext not in ['.h5']:
        raise argparse.ArgumentTypeError('only .h5 format supported')
    return fname


def is_dataset(fname):  # check whether the dataset format is supported by the framework.
    _, ext = os.path.splitext(fname)
    if ext not in ['.mat']:
        raise argparse.ArgumentTypeError('only .mat format supported')
    return fname


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    n_pool = multiprocessing.cpu_count()  # the number of CPUs in the system

    parser = argparse.ArgumentParser(description='SafeDeep')
    parser.add_argument('--net_name', type=is_network,help='name of the NN')
    parser.add_argument('--dataset', type=is_dataset, default="./Dataset/patient_01.mat", help='dataset')
    parser.add_argument('--delta', type=float, default=0.005, help='perturbation')
    parser.add_argument('--use_lp', type=str2bool, default=True, help='whether to use lp or not')
    args = parser.parse_args()

    net_name = args.net_name
    delta = args.delta  # perturbation
    flag_lp = args.use_lp  # layer-by-layer bound tightening
    dataset = scipy.io.loadmat(str(args.dataset))
    data = dataset['dataset']
    print("Dataset has ", np.shape(data)[0], " samples.")
    input_shape = np.shape(data)[1]
    lbl = dataset['label']

    modelNN = tf.keras.models.load_model(str(net_name))  # load model
    print("score is ", modelNN.evaluate(data, lbl, verbose=0))
    W = modelNN.get_weights()  # get weights
    modelNN.summary()  # print model summary

    n_layers = int(len(W) / 2) + 1
    n_neu = {}
    n_neu_cum = {}
    for k in range(n_layers):
        if k == 0:
            n_neu[k] = [np.shape(W[0])[0]]
            n_neu_cum[k] = [np.shape(W[0])[0]]
        elif k == n_layers - 1:
            n_neu[k] = [np.shape(W[-1])[0]]
            n_neu_cum[k] = [n_neu_cum[k - 1][-1] + np.sum(n_neu[k])]
        else:
            n_neu[k] = [np.shape(W[2 * k])[0], np.shape(W[2 * k])[0]]
            n_neu_cum[k] = [n_neu_cum[k - 1][-1] + np.shape(W[2 * k])[0], n_neu_cum[k - 1][-1] + np.sum(n_neu[k])]

    cc_data = 0  # correctly classified samples
    verified = 0  # robust samples
    num_samples = np.shape(lbl)[0]

    s_all = []

    t_all = TicToc()
    t = TicToc()
    t_all.tic()

    for i in range(num_samples):
        prd = modelNN.predict(np.reshape(data[i], [1, -1]))[0]
        t.tic()
        if ((lbl[i][0] == 0) & (prd[0] > prd[1])) or ((lbl[i][0] == 1) & (prd[0] < prd[1])):
            cc_data += 1

            lower_n = {}
            upper_n = {}
            ind_s_all = {}
            relation = {}

            cnt = data[i]  # data
            pert = delta * np.ones_like(cnt)  # perturbation

            mode_rng_calc = np.ones((n_layers - 1))  # 0: perturbation 1: simple min max calc 2: linear programming
            mode_rng_calc[0] = 0

            for s in range(1, n_layers - 1):
                if s > 1:
                    mode_rng_calc[s] = 2
                for k in range(n_layers - 1):
                    if s > 1 and k < s:
                        continue
                    if mode_rng_calc[k] == 0:
                        lower_n[0] = np.reshape(cnt - pert, [-1, 1])  # calculate lower bound of input neurons
                        upper_n[0] = np.reshape(cnt + pert, [-1, 1])  # calculate upper bound of input neurons
                    elif mode_rng_calc[k] == 1:
                        # calculate lower and upper bounds of all neurons by propagating through the network
                        ll, uu = bound_calc(W[2 * (k - 1)], W[2 * (k - 1) + 1], lower_n[k - 1], upper_n[k - 1])
                        lower_n[k] = ll
                        upper_n[k] = uu
                        relation[k] = (-1 * np.minimum(lower_n[k][:, 0], 0) * np.maximum(upper_n[k][:, 0], 0)) / 2
                        # relation helps to find undetermined neurons
                    else:
                        if flag_lp:  # check for layer-by-layer bound tightening
                            ll = np.copy(lower_n[k])
                            uu = np.copy(upper_n[k])
                            global W_tmp
                            W_tmp = W.copy()
                            W_tmp = W_tmp[:2 * k]
                            with Pool(n_pool) as p:  # create a multiprocessing Pool to calculate lower and upper bounds
                                # of all neurons of a layer in parallel
                                lower_tuple = p.starmap(gurobi_par_l, zip(range(n_neu[k][0]), repeat(relation[k]),
                                                                          repeat(lower_n), repeat(upper_n),
                                                                          repeat(W_tmp)))
                                upper_tuple = p.starmap(gurobi_par_u, zip(range(n_neu[k][0]), repeat(relation[k]),
                                                                          repeat(lower_n), repeat(upper_n),
                                                                          repeat(W_tmp)))
                            for i_t in range(n_neu[k][0]):
                                if lower_tuple[i_t] is not None:
                                    ll[i_t] = lower_tuple[i_t]
                            for i_t in range(n_neu[k][0]):
                                if lower_tuple[i_t] is not None:
                                    uu[i_t] = upper_tuple[i_t]
                            lower_n[k] = ll  # update lower bound
                            upper_n[k] = uu  # update upper bound

                if (lbl[i][0] == 0) & (prd[0] > prd[1]):
                    status, low_val = gb_opt(W, lower_n, upper_n, flag_min=True, flag_rev=False)
                elif (lbl[i][0] == 1) & (prd[0] < prd[1]):
                    status, low_val = gb_opt(W, lower_n, upper_n, flag_min=True, flag_rev=True)

                if low_val > 0:  # This means that the neural network is robust against the perturbed input
                    verified += 1
                    s_all.append(s)
                    print("break", "   Lower ", low_val)
                    break
                else:
                    print("can not verify with ", s, mode_rng_calc, "   Lower ", low_val)

            spam = t.tocvalue()
            hist, bins = np.histogram(s_all, bins=np.arange(0.5, n_layers - 0.5))
            print("Number ", i + 1, "-  Correct ", cc_data, "-  Verified ", verified, "State ", s, "hist_S ", hist,
                  "-  ElapsedTime ", spam)

        else:
            # This sample is not classified correctly,so its robustness has not been checked.
            hist, bins = np.histogram(s_all, bins=np.arange(0.5, n_layers - 0.5))
            print("Number ", i + 1, "-  Correct ", cc_data, "-  Verified ", verified, "hist_S ", hist)

    spam_all = t_all.tocvalue()

    print("Total ", i + 1, "-  Correct ", cc_data, "-  Verified ", verified, ", hist_S ", hist,
          "-  ElapsedTimeALL ", spam_all)
