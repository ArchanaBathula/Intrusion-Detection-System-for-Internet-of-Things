import numpy as np

def find_string(l1,s):
    matched_indexes = []
    i = 0
    length = len(l1)

    while i < length:
        if s == l1[i]:
            matched_indexes.append(i)
        i += 1
    return np.asarray(matched_indexes)

def normalize(data):
    b = 1
    a = 0
    norm_sam = None
    for k in range(data.shape[1]):
        d = data[:, k]
        ind = find_string(d, 'Infinity')
        if ind.any(): d[ind] = 100
        d = np.nan_to_num(d)
        mx = max(d)
        mn = min(d)
        if mx == mn:
            mx = 1
        dd = np.zeros((len(d),1))
        for i in range(len(d)):
            dd[i] = ((b - a) * ((d[i] - mn) / (mx - mn))) + a
        if norm_sam is None:
            norm_sam = dd
        else:
            norm_sam = np.concatenate(([norm_sam, dd]), axis=1)
    return norm_sam