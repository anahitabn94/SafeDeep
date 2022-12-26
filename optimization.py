# This function is called to check the robustness of the neural network.
# It is also used to calculate tighter bounds for each neuron.
# When flag_min is true, it calculates lower bound by minimizing the optimization function and it maximized the
# optimization function when flag_min is false.
# Flag_rev is used for the last layer for robustness verification.

import numpy as np
from gurobipy import Model, GRB, quicksum


def gb_opt(W, lower_n, upper_n, flag_min=True, flag_rev=False):
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
    n_neurons = np.sum([np.sum(n_neu[k]) for k in range(n_layers)])

    model = Model()  # Gurobi model
    variables = model.addVars(int(n_neurons),lb=-1 * float('inf'), name="variables")  # create variables

    model.Params.LogToConsole = 0  # suppress all output console
    model.Params.OutputFlag = 0  # suppress all output console

    for k in range(n_layers):
        if k == 0:  # lower and upper bounds of the first layer's neurons
            for jj in range(n_neu[k][0]):
                model.addConstr(variables[jj] >= lower_n[k][jj])
                model.addConstr(variables[jj] <= upper_n[k][jj])
        if 0 < k < n_layers - 1:  # equations and inequalities for the neurons of the hidden layers
            for m in range(n_neu[k][0]):
                ind_m = n_neu_cum[k][0] - n_neu[k][0] + m
                model.addConstr(quicksum(
                    W[2 * (k - 1)][z][m] * variables[n_neu_cum[k - 1][-1] - n_neu[k - 1][-1] + z] for z in
                    range(n_neu[k - 1][-1])) - variables[ind_m] == -1 * W[2 * (k - 1) + 1][m])
                ind_j = n_neu_cum[k][-1] - n_neu[k][-1] + m
                if lower_n[k][m][0] >= 0:  # neuron is always active
                    model.addConstr(variables[ind_j] == variables[ind_m])
                elif upper_n[k][m][0] < 0:  # neuron is always inactive
                    model.addConstr(variables[ind_j] == 0)
                else:  # neuron is undetermined
                    model.addConstr(variables[ind_j] >= 0)
                    model.addConstr(variables[ind_j] - variables[ind_m] >= 0)
                    model.addConstr(variables[ind_j] - upper_n[k][m][0] * (variables[ind_m] - lower_n[k][m][0]) /
                                    (upper_n[k][m][0] - lower_n[k][m][0]) <= 0)
        if k == n_layers - 1:  # equations for the neurons of the output layer
            for m in range(n_neu[k][0]):
                ind_m = n_neu_cum[k][0] - n_neu[k][0] + m
                model.addConstr(quicksum(
                    W[2 * (k - 1)][z][m] * variables[n_neu_cum[k - 1][-1] - n_neu[k - 1][-1] + z] for z in
                    range(n_neu[k - 1][-1])) - variables[ind_m] == -1 * W[2 * (k - 1) + 1][m])

    if flag_min:
        if n_neu[n_layers - 1][-1] == 1:  # calculate lower bound of a neuron
            ind0 = n_neu_cum[n_layers - 1][-1] - n_neu[n_layers - 1][-1]
            model.setObjective(variables[ind0], GRB.MINIMIZE)
        else:  # check robustness
            ind0 = n_neu_cum[n_layers - 1][-1] - n_neu[n_layers - 1][-1] + 0
            ind1 = n_neu_cum[n_layers - 1][-1] - n_neu[n_layers - 1][-1] + 1
            if not flag_rev:
                model.setObjective(variables[ind0] - variables[ind1], GRB.MINIMIZE)
            else:
                model.setObjective(variables[ind1] - variables[ind0], GRB.MINIMIZE)
    else:  # calculate upper bound of a neuron
        ind0 = n_neu_cum[n_layers - 1][-1] - n_neu[n_layers - 1][-1]
        model.setObjective(variables[ind0], GRB.MAXIMIZE)

    model.optimize()

    def print_solution():
        if model.status == GRB.OPTIMAL:
            stat = "optimal"
            val = model.ObjVal
        else:
            print('Model status: ', model.status)
            stat = "No solution"
            val = -1
        return stat, val

    status, value = print_solution()  # It has to be always optimal.

    return status, value
