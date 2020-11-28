import numpy as np
import random as rn
import time
import numpy.matlib
from function_Eval import feval

def PSO(val, objfun, x_min ,x_max, itermax, Feat,Tar,test_dat,test_tar):
    N, D = val.shape[0], val.shape[1]

    # Initialization of PSO parameters
    c1 = 2; c2 = 2
    wmax = 0.9; wmin = 0.1
    w = np.zeros((itermax, 1))
    for iter in range(itermax):
        w[iter] = wmax - ((wmax - wmin) / itermax) * iter  # Inertia weight update

    m = x_min[0]; n = x_max[0]
    q = (n - m) / (D * 2)
    Ki = 1

    # Random initialization of position and velocity
    x = val
    v = q* np.random.rand(N, D)

    f = feval(objfun, x, Feat,Tar,test_dat,test_tar)

    fgbest = min(f); igbest = np.where(min(f) == fgbest)
    gbest = x[igbest,:]
    pbest = x; fpbest = f

    fbst = np.zeros((itermax, 1))
    ct = time.time()

    # Iterate
    for it in range(itermax):

        # Update velocities and position
        v = w[it] * v +c1 * rn.random() * (pbest - x)+c2 * rn.random()
        x= x+v

        for mi in range(N):
            for mj in range(D):
                if x[mi, mj] < x_min[mi, mj]:
                    x[mi, mj] = x_min[mi, mj]
                else:
                    if x[mi, mj] > x_max[mi, mj]:
                        x[mi, mj] = x_max[mi, mj]

        f = feval(objfun, x, Feat,Tar,test_dat,test_tar)

        # Find global best and Particle best
        minf = min(f); iminf = np.where(min(f) == minf)
        if minf <= fgbest:
            fgbest = minf; gbest = x[iminf,:]
            best_sub = x[iminf,:]
            fbst[it] = minf
        else:
            fbst[it] = fgbest
            best_sub = gbest

        inewpb = np.where(f <= fpbest)
        pbest[inewpb,:] = x[inewpb[1],:]
        fpbest[inewpb] = f[inewpb]
    ct = time.time() - ct
    best_fit = fbst[itermax-1]

    return best_fit, fbst, best_sub, ct