import numpy as np
import numpy.matlib
from random import uniform
from Evaluation import evaln
from GWO import GWO
from Neural_Network import train_nn
from Results import plot_results
from Normalize import normalize
from PSO import PSO
from SHO import SHO
from data_augmentation import dataset1
from data_augmentation import dataset2
from data_augmentation import dataset3

### dataread
data1 = dataset1('.\KDD_cup99.xlsx')
data2 = dataset2('.\IOT_Intrusion_Dataset.xlsx')
data3 = dataset3('.\Third_Dataset.xlsx')

data = []
data.append(data1)
data.append(data2)
data.append(data3)

###  -------------------------------------  Optimization ------------------------------------------------------
an = 0
if an == 1:
    solns = []
    for i in range(0, 3):
        feat = normalize(data[i][:, 0:data[i].shape[1] - 2])
        tar = data[i][:, data[i].shape[1] - 1]

        per = round(feat.shape[0] * 0.70) # 70% of learning
        train_data = feat[0:per - 1, :]
        train_target = tar[0:per - 1]
        test_data = feat[per:per + data[i].shape[0] - 1, :]
        test_target = tar[per:per + data[i].shape[0] - 1]

        Npop = 10
        Ch_len = 1
        xmin = np.matlib.repmat(5,Npop,1)
        xmax = np.matlib.repmat(255, Npop, 1)
        initsol = np.zeros((xmax.shape))
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
        fname = 'objfun'
        Max_iter = 25

        print("PSO...")
        [bestfit, fitness, bestsol1, time] = PSO(initsol, fname, xmin, xmax, Max_iter, train_data, train_target, test_data, test_target)  # PSO

        print("GWO...")
        [bestfit, fitness, bestsol2, time] = GWO(initsol, fname, xmin, xmax, Max_iter, train_data, train_target, test_data, test_target)  # GWO

        print("SHO...")
        [bestfit, fitness, bestsol3, time] = SHO(initsol, fname, xmin, xmax, Max_iter, train_data, train_target, test_data, test_target)  # SHO

        bestsol = [bestsol1, bestsol2, bestsol3]
        solns.append(bestsol)
    np.save('solns.npy',solns)
else:
    solns = np.load('solns.npy', allow_pickle=True)



#####----------------------------------------------- Classification -----------------------------------------------
an = 0
if an == 1:
    Eval_all = []
    for i in range(0, 3):   # For all dataset
        print(i)
        feat = normalize(data[i][:, 0:data[i].shape[1] - 2])
        tar = data[i][:, data[i].shape[1] - 1]
        pn = [0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
        Eval_out = []
        for p in range(6):  ### For all learning percentage
            print(p)
            per = round(feat.shape[0] * pn[p])  # 70% of learning
            train_data = feat[0:per - 1, :]
            train_target = tar[0:per - 1]
            test_data = feat[per:per + data[i].shape[0] - 1, :]
            test_target = tar[per:per + data[i].shape[0] - 1]
            act = test_target

            EVAL = np.zeros((4,14))
            for n in range(4):
                print(n)
                if n == 0:
                    so = 10
                else:
                    sol = solns[i][n-1]
                    so = sol[0]
                pred, net = train_nn(train_data, train_target, test_data, round(so))
                act.astype(bool)
                pred.astype(bool)
                EVAL[n,:] = evaln([pred], [act])
            Eval_out.append(EVAL)
        Eval_all.append(Eval_out)
    np.save('Eval_all.npy', Eval_all)

plot_results()
