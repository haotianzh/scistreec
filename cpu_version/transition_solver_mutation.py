"""
No Mutation
"""
import numpy as np 
import sympy as sp
import cupy as cp
import pandas as pd

CN_MAX = 4
CN_MIN = 2
LAMBDA_C = 1
LAMBDA_S = 1
LAMBDA_T = 4
CNV_TIMEOUT  = LAMBDA_C / (LAMBDA_C + LAMBDA_T)
CNV_SNV = LAMBDA_C / (LAMBDA_C + LAMBDA_S)
N = int((CN_MAX-CN_MIN+1) * (CN_MAX+CN_MIN+2) / 2)

def index(i, j, p ,q):
    # use yang's tri indexing
    i1 = (i+j) * (i+j+1) / 2 + i - int((CN_MIN) * (CN_MIN + 1) / 2)
    i2 = (p+q) * (p+q+1) / 2 + p - int((CN_MIN) * (CN_MIN + 1) / 2)
    return int(i1*N + i2)


def allow_increase(g0, g1):
    return g0 + g1 < CN_MAX and g0 + g1 >= 1

def allow_decrease(g0, g1):
    return g0 + g1 > CN_MIN

def valid(g0, g1):
    return CN_MIN <= g0 + g1 <= CN_MAX

def identity(g0, g1, g0_, g1_):
    return g0 == g0_ and g1 == g1_

def equation_mutation():
    eqs = []
    vals = []
    for g0 in range(CN_MAX+1):
        for g1 in range(CN_MAX+1):
            if not valid(g0, g1):
                continue
            for g0_ in range(CN_MAX+1):
                for g1_ in range(CN_MAX+1):
                    if not valid(g0_, g1_):
                        continue
                    eq = np.zeros(N*N, dtype=float)
                    if g0 == 0 and g1 == 0:
                        eq[index(g0, g1, g0_, g1_)] = 1
                        val = identity(g0, g1, g0_, g1_)
                        eqs.append(eq)
                        vals.append(val)
                        continue
                    p = allow_increase(g0, g1) / (allow_increase(g0, g1) + allow_decrease(g0, g1))
                    eq[index(g0, g1, g0_, g1_)] = -1
                    if valid(g0+1, g1):
                        eq[index(g0+1, g1, g0_, g1_)] = p * CNV_TIMEOUT * (g0 / (g0+g1))
                    if valid(g0, g1+1):
                        eq[index(g0, g1+1, g0_, g1_)] = p * CNV_TIMEOUT * (g1 / (g0+g1))
                    if valid(g0-1, g1):
                        eq[index(g0-1, g1, g0_, g1_)] = (1-p) * CNV_TIMEOUT * (g0 / (g0+g1))
                    if valid(g0, g1-1):
                        eq[index(g0, g1-1, g0_, g1_)] = (1-p) * CNV_TIMEOUT * (g1 / (g0+g1))
                    val = -identity(g0, g1, g0_, g1_) * (1 - CNV_TIMEOUT)
                    eqs.append(eq)
                    vals.append(val)
    return np.array(eqs), np.array(vals)


def solve_mutation():
    eqs, vals = equation_mutation()
    print('P(CNV before TIMEOUT): ', CNV_TIMEOUT)
    x = np.linalg.solve(eqs, vals)
    x = np.abs(np.round(x, 2))
    for g0 in range(CN_MAX+1):
        for g1 in range(CN_MAX+1):
            if not valid(g0, g1):
                continue
            for g0_ in range(CN_MAX+1):
                for g1_ in range(CN_MAX+1):
                    if not valid(g0_, g1_):
                        continue
                    print(f'({g0},{g1})->({g0_},{g1_}) = ', x[index(g0, g1, g0_, g1_)])

if __name__ == "__main__":
    solve_mutation()
    # print(index(0, 0, 3, 1))
    
                    
                    