import numpy as np
from gurobipy import Model, GRB, quicksum
from utils import get_layer_shapes


def gb_opt(W, lower_n, upper_n, minimize=True, reverse=False):
    """MILP-based bound tightening / robustness verification with Gurobi."""
    n_layers, n_neu, n_neu_cum = get_layer_shapes(W)
    n_neurons = sum(np.sum(n_neu[k]) for k in range(n_layers))

    model = Model()
    model.Params.LogToConsole = 0
    model.Params.OutputFlag = 0

    variables = model.addVars(int(n_neurons), lb=-GRB.INFINITY, name="x")

    # Build constraints layer by layer
    for k in range(n_layers):
        if k == 0:  # input bounds
            for j in range(n_neu[k][0]):
                model.addConstr(variables[j] >= lower_n[k][j])
                model.addConstr(variables[j] <= upper_n[k][j])

        elif 0 < k < n_layers - 1:  # hidden layers
            for m in range(n_neu[k][0]):
                ind_m = n_neu_cum[k][0] - n_neu[k][0] + m  # pre-activation
                # linear relation
                model.addConstr(quicksum(
                    W[2*(k-1)][z][m] *
                    variables[n_neu_cum[k-1][-1] - n_neu[k-1][-1] + z]
                    for z in range(n_neu[k-1][-1])
                ) - variables[ind_m] == -W[2*(k-1)+1][m])

                ind_j = n_neu_cum[k][-1] - n_neu[k][-1] + m  # post-activation
                lo, up = lower_n[k][m][0], upper_n[k][m][0]

                if lo >= 0:  # always active
                    model.addConstr(variables[ind_j] == variables[ind_m])
                elif up < 0:  # always inactive
                    model.addConstr(variables[ind_j] == 0)
                else:  # ambiguous
                    model.addConstr(variables[ind_j] >= 0)
                    model.addConstr(variables[ind_j] - variables[ind_m] >= 0)
                    slope = up / (up - lo)
                    model.addConstr(variables[ind_j] <= slope * (variables[ind_m] - lo))
        else:  # output layer
            for m in range(n_neu[k][0]):
                ind_m = n_neu_cum[k][0] - n_neu[k][0] + m
                model.addConstr(quicksum(
                    W[2*(k-1)][z][m] *
                    variables[n_neu_cum[k-1][-1] - n_neu[k-1][-1] + z]
                    for z in range(n_neu[k-1][-1])
                ) - variables[ind_m] == -W[2*(k-1)+1][m])

    # --- Objective ---
    if minimize:
        if n_neu[n_layers - 1][-1] == 1:
            ind0 = n_neu_cum[n_layers - 1][-1] - 1
            model.setObjective(variables[ind0], GRB.MINIMIZE)
        else:
            ind0 = n_neu_cum[n_layers - 1][-1] - n_neu[n_layers - 1][-1]
            ind1 = ind0 + 1
            if not reverse:
                model.setObjective(variables[ind0] - variables[ind1], GRB.MINIMIZE)
            else:
                model.setObjective(variables[ind1] - variables[ind0], GRB.MINIMIZE)
    else:
        ind0 = n_neu_cum[n_layers - 1][-1] - n_neu[n_layers - 1][-1]
        model.setObjective(variables[ind0], GRB.MAXIMIZE)

    # --- Solve ---
    model.optimize()
    if model.status == GRB.OPTIMAL:
        return "optimal", model.ObjVal
    return "infeasible", None
