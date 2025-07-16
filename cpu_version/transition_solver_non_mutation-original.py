import numpy as np 
import sympy as sp
import cupy as cp
import pandas as pd


def index(i, j, p ,q, n):
    # use yang's tri indexing
    i1 = (i+j) * (i+j+1) / 2 + i
    i2 = (p+q) * (p+q+1) / 2 + p
    return int(i1*n + i2)

# for no mutation
def solve(cn, lambda_s, lambda_c, lambda_e):
    tc = lambda_c / (lambda_c + lambda_e)
    tc = 0.2
    n = int((cn+1) * (cn+2) / 2)
    # print('n', n)
    mat = []
    vals = []
    for i in range(cn+1):
        for j in range(cn+1):
            if i + j > cn:
                continue
            for p in range(cn+1):
                for q in range(cn+1):
                    if p + q > cn:
                        continue
                    vec = np.zeros([n*n], dtype=float)
                    ind = index(i, j, p, q, n)
                    val = 0
                    vec[ind] = 1
                    # self transition
                    if i == p and j == q:
                        for p1 in range(cn+1):
                            for q1 in range(cn+1):
                                if p1 + q1 > cn:
                                    continue
                                vec[index(i, j, p1, q1, n)] = 1
                                val = 1
                        mat.append(vec)
                        vals.append(val)
                        continue
                    if i >= 1 and j >= 1:
                        vec[ind] = 1
                        if i+j == cn:
                            vec[index(i-1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i, j-1, p, q, n)] = -j*tc / 2 / (i+j)
                        elif i < cn and j < cn:
                            vec[index(i+1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i-1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i, j+1, p, q, n)] = -j*tc / 2 / (i+j)
                            vec[index(i, j-1, p, q, n)] = -j*tc / 2 / (i+j)
                        elif i == cn and j < cn:
                            vec[index(i-1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i, j+1, p, q, n)] = -j*tc / 2 / (i+j)
                            vec[index(i, j-1, p, q, n)] = -j*tc / 2 / (i+j)
                        elif i < cn and j == cn:
                            vec[index(i+1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i-1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i, j-1, p, q, n)] = -j*tc / 2 / (i+j)
                        else:
                            vec[index(i-1, j, p, q, n)] = -i*tc / 2 / (i+j)
                            vec[index(i, j-1, p, q, n)] = -j*tc / 2 / (i+j)
                    if i >= 1 and j == 0:
                        if q >= 1:    
                            vec[index(i, j, p, q, n)] = 1
                        else:
                            if i < cn:
                                vec[index(i+1, j, p, q, n)] = -tc / 2
                                vec[index(i-1, j ,p, q, n)] = -tc / 2
                            else:
                                vec[index(i-1, j ,p, q, n)] = -tc / 2
                    if i == 0 and j >= 1:
                        if p >= 1:
                            vec[index(i, j, p, q, n)] = 1
                        else:
                            if j < cn:
                                vec[index(i, j+1, p, q, n)] = -tc / 2
                                vec[index(i, j-1, p, q, n)] = -tc / 2
                            else:
                                vec[index(i, j-1, p, q, n)] = -tc / 2
                    if i == 0 and j == 0:
                        if p >= 1 or q >= 1:
                            vec[index(i, j, p, q, n)] = 1
                        else:
                            vec[index(i, j, p, q, n)] = 1
                            val = 1
                    mat.append(vec)
                    vals.append(val)

    return np.array(mat), np.array(vals)


if __name__ == "__main__":
    k = 2
    mat, val = solve(k, 1, 1, 1)
    print(mat.shape, val.shape)
    print(mat)
    print(val)
    x = np.linalg.lstsq(mat, val, rcond=-1)
    # x = np.linalg.solve(mat, val)

    g0 = []
    g1 = []
    p0 = []
    p1 = []
    prob = []
    for i in range(k+1):
        for j in range(k+1):
            if i + j > k:
                continue
            for p in range(k+1):
                for q in range(k+1):
                    if p + q > k:
                        continue
                    ind = index(i, j, p, q, int((k+1)*(k+2)/2))
                    g0.append(i)
                    g1.append(j)
                    p0.append(p)
                    p1.append(q)
                    prob.append(np.round(x[0][ind], 2))
    data = {'g0': g0, 'g1': g1, 'p0': p0, 'p1': p1, 'prob': prob}
    df = pd.DataFrame(data)
    print(df)
    df.to_csv(f'k_{k}_prob.csv', index=False)

    # print(mat[[6, 0, 1, 2, 7, 3, 4, 5, 8]])
    print(mat)
    print(sp.Matrix(mat).T.rref())