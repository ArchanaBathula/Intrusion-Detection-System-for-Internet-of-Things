import numpy as np
import pandas as pd

def find_string(l1,s):
    matched_indexes = []
    i = 0
    for i in range(len(l1)):
        if s == l1[i]:
            matched_indexes.append(i)
    return np.asarray(matched_indexes)

def dataset1(filename):
    an = 0
    if an == 1:
        wb = pd.ExcelFile(filename)  # Read Excel file
        df1 = wb.parse('Sheet1')
        data = df1.values  # Get the values of sheet1
        for n1 in range(data.shape[1]):
            if isinstance(data[0, n1], str):
                if n1 == data.shape[1] - 1:
                    val = np.ones((data.shape[0]))
                    ind = find_string(data[:, n1], 'normal')
                    if ind.any(): val[ind] = int(0)
                    data[:, n1] = val
                else:
                    u = np.unique(data[:, n1])
                    for n2 in range(len(u)):
                        ind = find_string(data[:, n1], u[n2])
                        data[ind, n1] = n2
        np.save('data.npy', data)
    else:
        data = np.load('data.npy', allow_pickle=True)

    return data

def dataset2(filename):
    an = 0
    if an == 1:
        excel = pd.ExcelFile(filename)
        sheet = excel.parse('Sheet1')
        data = sheet.values
        for col in range(data.shape[1]):
            if isinstance(data[0, col], str):
                """value = np.ones(data.shape[0])
                uniq = np.unique(data[1:, col])
                for ind_uniq in range(len(uniq)):
                    index = find_string(data[1:, col], uniq[ind_uniq])
                    for i in range(len(index)):
                        value[index] = ind_uniq"""

                if col == data.shape[1] - 1:
                    value = np.ones(data.shape[0])
                    index = find_string(data[:, col], 'Normal')
                    for ind in range(len(index)):
                        value[index] = int(0)
                    data[:, col] = value
                else:
                    uniq = np.unique(data[:, col])
                    for ind_uniq in range(len(uniq)):
                        index = find_string(data[:, col], uniq[ind_uniq])
                        for ind in range(len(index)):
                            data[index, col] = ind_uniq
        np.save('data2.npy', data)
    else:
        data = np.load('data2.npy', allow_pickle=True)
    return data

def dataset3(filename):
    an = 0
    if an == 1:
        excel = pd.ExcelFile(filename)
        sheet = excel.parse('Sheet1')
        data = sheet.values
        for col in range(data.shape[1]):
            if isinstance(data[0, col], str):
                if col == data.shape[1] - 1:
                    value = np.ones(data.shape[0])
                    index = find_string(data[:, col], 'Normal')
                    for ind in range(len(index)):
                        value[index] = int(0)
                    data[:, col] = value
                else:
                    uniq = np.unique(data[:, col])
                    for ind_uniq in range(len(uniq)):
                        index = find_string(data[:, col], uniq[ind_uniq])
                        for ind in range(len(index)):
                            data[index, col] = ind_uniq
        np.save('data3.npy', data)
    else:
        data = np.load('data3.npy', allow_pickle=True)

    return data