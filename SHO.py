import numpy as np
import random as rn
import time
from function_Eval import feval

def SHO(Positions,fobj,Lb,Ub,Max_iter,val_in,val_tar,test_dat,test_tar):
    fitness = 0
    N, dim = Positions.shape[0], Positions.shape[1]
    ub = Ub[1, :]
    lb = Lb[1, :]

    # initialize alpha, beta, and delta_pos
    Alpha_pos = np.zeros((dim, 1))
    Alpha_score = float('inf')

    Convergence_curve = np.zeros((Max_iter + 1, 1))
    l = 0
    ct = time.time()
    new_fitness = []
    while l <= Max_iter:
        for i in range(N):
            # Return back the search agents that go beyond the boundaries of the search space
            Flag4ub = Positions[i, :] > ub
            Flag4lb = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (~(Flag4ub + Flag4lb))) + ub * Flag4ub + lb * Flag4lb

            # Calculate objective function for each search agent
            new_fitness.append(feval(fobj, Positions[i, :], val_in, val_tar, test_dat, test_tar))

            #  Update the leader
            if new_fitness[i] < Alpha_score:
                Alpha_score = new_fitness[i]  # Update alpha
                Alpha_pos = Positions[i, :]
        a = 5 - l * ((5) / Max_iter)

        if l == 0:
            indn = np.zeros(dim, 1)
        else:
            indn = new_fitness<fitness

        for i in range(N):
            if indn == 1:
                h = rn.randint(1, N-5)
            else:
                h = 1
            for j in range(dim):
                X1 = []
                for c in range(h):
                    r1 = rn.random()
                    r2 = rn.random()

                    A1 = 2 * a * r1 - a
                    C1 = 2 * r2

                    D_alpha = abs(C1 * Alpha_pos[c, j] - Positions[i, j])
                    X1[c] = Alpha_pos[c, j] - A1 * D_alpha
                Positions[i, j] = sum(X1) / dim
        Convergence_curve[l] = Alpha_score
        l = l + 1
        fitness = new_fitness
    best_fit = Convergence_curve[Max_iter-1]
    ct = time.time() - ct
    return best_fit, Convergence_curve, Alpha_pos, ct