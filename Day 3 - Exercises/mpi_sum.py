# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:11:21 2024

@author: enls0001
"""

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
numbers = np.random.random(size)
total = np.zeros(1)

number = np.array(numbers[rank])

comm.Reduce(number,total,op=MPI.SUM,root=0)

print('Rank {} has the value {}'.format(rank,number))

if rank == 0:
    print('Sum of values is {}'.format(total[0]))
