import numpy as np
from Neural_Network import train_nn
from Evaluation import evaln

def objfun(soln, train_data, train_target, test_data, test_target):
    if soln.ndim == 2:
        dim = soln.shape[1]   # length of soln
        v = soln.shape[0]  # No. of soln
        fitn = np.zeros((soln.shape[0], 1))
    else:
        dim = soln.shape[0]; v = 1
        fitn = np.zeros((1, 1))

    for i in range(v):
        soln = np.array(soln)

        if soln.ndim == 2:
            sol = np.round(soln[i,:])
        else:
            sol = np.round(soln)

        act = test_target
        pred, net = train_nn(train_data, train_target, test_data, int(sol))  ## ANN training
        act.astype(bool)
        pred.astype(bool)
        Eval = evaln([pred], [act])
        fitn[i] = 1 / (Eval[4]+Eval[7])

    return fitn