#!/usr/bin/python

# Program to multiply two matrices using nested loops
import random
from line_profiler import LineProfiler
import numpy as np

N = 250

def matmult(N):
    # NxN matrix
    X = []
    for i in range(N):
        X.append([random.randint(0,100) for r in range(N)])
    X = np.array(X)
    
    # Nx(N+1) matrix
    Y = []
    for i in range(N):
        Y.append([random.randint(0,100) for r in range(N+1)])
    Y = np.array(Y)
    
    result = np.matmul(X,Y)
    
    # result is Nx(N+1)
    # result = []
    # for i in range(N):
    #     result.append([0] * (N+1))
    # result = np.array(result)
    
    # # iterate through rows of X
    # for i in range(len(X)):
    #     # iterate through columns of Y
    #     for j in range(len(Y[0])):
    #         # iterate through rows of Y
    #         for k in range(len(Y)):
    #             result[i][j] += X[i][k] * Y[k][j]
    
    for r in range(0,result.shape[0]):
        print(result[r,:])

lp = LineProfiler()
lp_wrapper = lp(matmult)
lp_wrapper(N)
lp.print_stats()