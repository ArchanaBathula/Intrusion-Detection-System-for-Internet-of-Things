import numpy as np
import random as rn
import time
from function_Eval import feval

def GWO(Positions,fobj,Lb,Ub,Max_iter,val_in,val_tar,test_dat,test_tar):
    N, dim = Positions.shape[0], Positions.shape[1]
    ub = Ub[1,:]
    lb = Lb[1,:]

    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros((dim, 1))
    Alpha_score = float('inf')

    Beta_pos = np.zeros((dim, 1))
    Beta_score = float('inf')

    Delta_pos = np.zeros((dim, 1))
    Delta_score = float('inf')

    Convergence_curve = np.zeros((Max_iter+1, 1))
    l = 0
    ct = time.time()

    while l <= Max_iter:
        for i in range(N):
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

            # Calculate objective function for each search agent
            fitness = feval(fobj, Positions[i, :],val_in,val_tar,test_dat,test_tar)

            #  Update the leader
            if fitness < Alpha_score:
                Alpha_score = fitness  # Update alpha
                Alpha_pos = Positions[i, :]

            if  Alpha_score < fitness < Beta_score:
                Beta_score = fitness
                Beta_pos = Positions[i, :]

            if Alpha_score < fitness < Beta_score:
                if fitness < Delta_score:
                   Delta_score = fitness # Update delta
                   Delta_pos = Positions[i, :]

        a = 2 - l * ((2) / Max_iter)  # a decreases linearly fron 2 to 0

        # Update the Position of search agents including omegas

        for i in range(N):
            for j in range(dim):
                r1 = rn.random()
                r2 = rn.random()

                A1 = 2 * a * r1 - a
                C1 = 2 * r2

                D_alpha = abs(C1 * Alpha_pos[j] - Positions[i, j])
                X1 = Alpha_pos[j] - A1 * D_alpha

                r1 = rn.random()
                r2 = rn.random()

                A2 = 2 * a * r1 - a
                C2 = 2 * r2

                D_beta = abs(C2 * Beta_pos[j] - Positions[i, j])
                X2 = Beta_pos[j] - A2 * D_beta

                r1 = rn.random()
                r2 = rn.random()

                A3 = 2 * a * r1 - a
                C3 = 2 * r2

                D_delta = abs(C3 * Delta_pos[j] - Positions[i, j])
                X3 = Delta_pos[j] - A3 * D_delta

                Positions[i, j] = (X1 + X2 + X3) / 3

        Convergence_curve[l] = Alpha_score
        l = l + 1

    best_fit = Convergence_curve[Max_iter-1]
    ct = time.time() - ct
    return best_fit, Convergence_curve, Alpha_pos, ct

