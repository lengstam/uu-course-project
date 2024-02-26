# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 13:39:35 2024

@author: enls0001
"""

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("Rank is {}".format(rank))