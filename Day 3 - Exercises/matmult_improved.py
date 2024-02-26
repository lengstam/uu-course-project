#!/usr/bin/python

# Program to multiply two matrices using nested loops
# import random
from line_profiler import LineProfiler
import numpy as np

N = 250

def matmult(N):
    # NxN matrix
    X = np.random.randint(0,100,[N,N])
    # X = []
    # for i in range(N):
    #     X.append([random.randint(0,100) for r in range(N)])
    # X = np.array(X)
    
    # Nx(N+1) matrix
    Y = np.random.randint(0,100,[N,N+1])
    # Y = []
    # for i in range(N):
    #     Y.append([random.randint(0,100) for r in range(N+1)])
    # Y = np.array(Y)
    
    result = np.matmul(X,Y)
    
    # for r in range(0,result.shape[0]):
    #     print(result[r,:])
    print(result)

lp = LineProfiler()
lp_wrapper = lp(matmult)
lp_wrapper(N)
lp.print_stats()