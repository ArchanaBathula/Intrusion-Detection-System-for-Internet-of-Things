import matplotlib.pyplot as plt
from prettytable import PrettyTable
import numpy as np

def plot_results():
    Eval = np.load('Eval_all.npy')

    row_head = ["Accuracy", "Sensitivity", "Specificity", "Precision", "FPR", "FNR", "NPV", "FDR", "F1_score", "MCC"]
    column_head = ["Term", "ANN", "PSO.ANN", "GWO.ANN", "SHO.ANN"]
    t1 = PrettyTable()
    t2 = PrettyTable()
    t3 = PrettyTable()
    t1.add_column("Term", row_head)
    t2.add_column("Term", row_head)
    t3.add_column("Term", row_head)


    print("----------------------------------- Tables -----------------------------------")
    print()

    ### First Dataset
    print("----------------------------------- DataSet 1 -----------------------------------")
    for i in range(len(Eval[0][4])):
        row = []
        for j in range(len(Eval[0][4][i]) - 4):
            row.append(Eval[0][4][i][j+4])
        t1.add_column(column_head[i+1], row)
    print(t1)
    print()

    ### Second Dataset
    print("----------------------------------- DataSet 2 -----------------------------------")
    for i in range(len(Eval[1][4])):
        row = []
        for j in range(len(Eval[1][4][i]) - 4):
            row.append(Eval[1][4][i][j+4])
        t2.add_column(column_head[i+1], row)
    print(t2)
    print()

    ### Third Dataset
    print("----------------------------------- DataSet 3 -----------------------------------")
    for i in range(len(Eval[2][4])):
        row = []
        for j in range(len(Eval[2][4][i]) - 4):
            row.append(Eval[2][4][i][j+4])
        t3.add_column(column_head[i+1], row)
    print(t3)
    print()

    ### First DataSet
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    pn = [35, 45, 55, 65, 75, 85]

    accur = np.zeros((4, 6))
    for i in range(len(Eval[0])):
        for j in range(len(Eval[0][i])):
            accur[j, i] = Eval[0][i][j][4]*100
    X = np.arange(6)
    ax.bar(X + 0.00, accur[0, :], color = 'b', width = 0.10, label='ANN [17]')
    ax.bar(X + 0.10, accur[1, :], color = 'g', width = 0.10, label='PSO.ANN [18]')
    ax.bar(X + 0.20, accur[2, :], color = 'r', width = 0.10, label='GWO.ANN [19]')
    ax.bar(X + 0.30, accur[3, :], color = 'k', width = 0.10, label='SHO.ANN')
    plt.xticks(X+0.10, (35, 45, 55, 65, 75, 85))
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Learning Percentage")
    plt.legend(loc = 4)
    path1 = "./Results/Result001.png"
    plt.savefig(path1)
    plt.show()

    ### Second DataSet
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    pn = [35, 45, 55, 65, 75, 85]

    accur = np.zeros((4, 6))
    for i in range(len(Eval[1])):
        for j in range(len(Eval[1][i])):
            accur[j, i] = Eval[1][i][j][4]*100
    X = np.arange(6)
    ax.bar(X + 0.00, accur[0, :], color = 'b', width = 0.10, label='ANN [17]')
    ax.bar(X + 0.10, accur[1, :], color = 'g', width = 0.10, label='PSO.ANN [18]')
    ax.bar(X + 0.20, accur[2, :], color = 'r', width = 0.10, label='GWO.ANN [19]')
    ax.bar(X + 0.30, accur[3, :], color = 'k', width = 0.10, label='SHO.ANN')
    plt.xticks(X+0.10, (35, 45, 55, 65, 75, 85))
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Learning Percentage")
    plt.legend(loc = 4)
    path1 = "./Results/Result002.png"
    plt.savefig(path1)
    plt.show()

    ### Third DataSet
    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    pn = [35, 45, 55, 65, 75, 85]

    accur = np.zeros((4, 6))
    for i in range(len(Eval[2])):
        for j in range(len(Eval[2][i])):
            accur[j, i] = Eval[2][i][j][4]*100
    X = np.arange(6)
    ax.bar(X + 0.00, accur[0, :], color = 'b', width = 0.10, label='ANN [17]')
    ax.bar(X + 0.10, accur[1, :], color = 'g', width = 0.10, label='PSO.ANN [18]')
    ax.bar(X + 0.20, accur[2, :], color = 'r', width = 0.10, label='GWO.ANN [19]')
    ax.bar(X + 0.30, accur[3, :], color = 'k', width = 0.10, label='SHO.ANN')
    plt.xticks(X+0.10, (35, 45, 55, 65, 75, 85))
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Learning Percentage")
    plt.legend(loc = 4)
    path1 = "./Results/Result003.png"
    plt.savefig(path1)
    plt.show()