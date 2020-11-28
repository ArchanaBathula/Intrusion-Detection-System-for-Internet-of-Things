import matplotlib.pyplot as plt
from tabulate import tabulate
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def plot_graph():
    for i in range(4):  # 4 testcases
        if i == 0:
            Error_eval = np.load('Error_eval1.npy')
            print('-----------------------Testcase: 1--------------------------------------')
        elif i == 1:
            Error_eval = np.load('Error_eval2.npy')
            print('-----------------------Testcase: 2--------------------------------------')
        elif i == 2:
            Error_eval = np.load('Error_eval3.npy')
            print('-----------------------Testcase: 3--------------------------------------')
        else:
            Error_eval = np.load('Error_eval4.npy')
            print('-----------------------Testcase: 4--------------------------------------')

        Ev = Error_eval[3]  # Only the learning percentage 75

        Eval = Ev[:,4:]  # Leave tp tn fp fn
        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Algorithm+Classifier Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        Ev1 = ['PSO','GWO','WOA','DHOA','Proposed']
        Ev1 = np.array(Ev1).reshape(-1,1)
        Ev = np.concatenate([Ev1, Eval[0:5,:]], axis=1)
        print('Accuracy  Sensitivity  Specificity  Precision  FPR  FNR  NPV  FDR   F1-Score  MCC')
        for n in range(Ev.shape[0]):
            print(Ev[n][0], Ev[n][1], Ev[n][2], Ev[n][3], Ev[n][4], Ev[n][5], Ev[n][6], Ev[n][7], Ev[n][8], Ev[n][9], Ev[n][10])


        print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Classifier Analysis %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
        Ev1 = ['Fuzzy', 'NN', 'SVM', 'KNN', 'Fuzzy+NN', 'Proposed']
        Ev1 = np.array(Ev1).reshape(-1, 1)
        Ev = np.concatenate([Ev1, Eval[[6,7,8,5,9,4], :]], axis=1)
        print('Accuracy  Sensitivity  Specificity  Precision  FPR  FNR  NPV  FDR   F1-Score  MCC')
        for n in range(Ev.shape[0]):
            print(Ev[n][0], Ev[n][1], Ev[n][2], Ev[n][3], Ev[n][4], Ev[n][5], Ev[n][6], Ev[n][7], Ev[n][8], Ev[n][9],
                  Ev[n][10])

        lnn = ['Accuracy',  'Sensitivity',  'Specificity',  'Precision',  'FPR',  'FNR',  'NPV',  'FDR',   'F1-Score',  'MCC']
        x = [35, 55, 65, 75, 85]
        val = np.zeros((5,5))
        vn = [0, 3, 4, 5, 8]
        for j in range(len(vn)):
            for k in range(5):
                Ev = Error_eval[k]
                Eval = Ev[:, 4:13]
                data = Eval[0:5,:]

                for m in range(5):
                    if vn[j] == 9:
                        val[m,k] = data[m,vn[j]]
                    else:
                        val[m, k] = data[m, vn[j]]*100

            plt.plot(x, val[0,:], color='black', linewidth=3, marker='o', markerfacecolor='blue', markersize=12, label = "PSO-FNN")
            plt.plot(x, val[1, :], color='black', linewidth=3, marker='o', markerfacecolor='red', markersize=12, label = "GWO-FNN")
            plt.plot(x, val[2, :], color='black', linewidth=3, marker='o', markerfacecolor='green', markersize=12, label = "WOA-FNN")
            plt.plot(x, val[3, :], color='black', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12, label = "DHOA-FNN")
            plt.plot(x, val[4, :], color='black', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12, label = "O-DHOA-FNN")
            plt.xlabel('Learning Percentage')
            plt.ylabel(lnn[vn[j]])
            plt.legend(loc="NorthEastOutside")
            path1 = "./Results/perfalg_%s-%s.png" % (i, j)
            plt.savefig(path1)
            plt.show()


        val = np.zeros((6, 5))
        for j in range(len(vn)):
            for k in range(5):
                Ev = Error_eval[k]
                Eval = Ev[:, 4:13]
                data = Eval[[6, 7, 8, 5, 9, 4], :]

                for m in range(6):
                    if vn[j] == 9:
                        val[m, k] = data[m, vn[j]]
                    else:
                        val[m, k] = data[m, vn[j]] * 100

            plt.plot(x, val[0, :], color='black', linewidth=3, marker='o', markerfacecolor='blue', markersize=12,
                     label="Fuzzy")
            plt.plot(x, val[1, :], color='black', linewidth=3, marker='o', markerfacecolor='red', markersize=12,
                     label="NN")
            plt.plot(x, val[2, :], color='black', linewidth=3, marker='o', markerfacecolor='green', markersize=12,
                     label="SVM")
            plt.plot(x, val[3, :], color='black', linewidth=3, marker='o', markerfacecolor='magenta', markersize=12,
                     label="KNN")
            plt.plot(x, val[4, :], color='black', linewidth=3, marker='o', markerfacecolor='yellow', markersize=12,
                     label="FNN")
            plt.plot(x, val[5, :], color='black', linewidth=3, marker='o', markerfacecolor='cyan', markersize=12,
                     label="O-DHOA-FNN")
            plt.xlabel('Learning Percentage')
            plt.ylabel(lnn[vn[j]])
            plt.legend(loc="NorthEastOutside")
            path1 = "./Results/perfcls_%s-%s.png" % (i, j)
            plt.savefig(path1)
            plt.show()







