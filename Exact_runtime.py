#!/usr/bin/env python
# coding: utf-8

# In[2]:


import netket as nk
import json
from qutip import *
import numpy as np
import time
import multiprocessing as mp
from collections import OrderedDict
from pickle import dump
import os
import matplotlib.pyplot as plt
import scipy
from matplotlib import gridspec
from functools import reduce
from functools import wraps
plt.style.use('seaborn')
from scipy.stats import norm


# In[3]:


def CSHam(N, B, Ak):
    # Make graph with of length N with no periodic boundary conditions
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=False)
    # Spin based Hilbert Space
    hilbertSpace = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
    # Define spin operators with \hbar set to 1
    sz = 0.5 * np.array([[1, 0], [0, -1]])
    sx = 0.5 * np.array([[0, 1], [1, 0]])
    sy = 0.5 * np.array([[0, -1j], [1j, 0]])
    operators = []
    sites = []
    # Central spin term
    operators.append((B * sz).tolist()) #array to list(ordered and changeable)
    sites.append([0])
    # Interaction term
    itOp = np.kron(sz, sz) + np.kron(sx, sx) + np.kron(sy, sy) #kronecker product here
    for i in range(N - 1):
        operators.append((Ak[i] * itOp).tolist())
        sites.append([0, (i+1)])  #pretty convoluted indexing, but ok
    # Create hamiltonian
    hamiltonian = nk.operator.LocalOperator(hilbertSpace, operators=operators, acting_on=sites, dtype=complex)
    #acting_on specifier necessary as this is a central spin model
    return hamiltonian, hilbertSpace


# In[4]:


#Wrapper to time functions
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ti = time.time()
        result = f(*args, **kw)
        tf = time.time()
        t = tf-ti
        return result, t
    return wrap


def averageOfList(num):
    sumOfNumbers = 0
    for t in num:
        sumOfNumbers = sumOfNumbers + t

    avg = sumOfNumbers / len(num)
    return avg


# In[5]:


#Lanczos algorithm, with only the ground state
@timing
def exactDiagonalization(hamiltonian):
    # Changes Hamiltonian to matrix form, where hamiltonian of interest is sparse in matrix form
    #haMatrix = hamiltonian.to_sparse()
    # Gets eigenvalues and vectors, where the built-in function uses 
    eigenValues, v = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)

    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors

#brute-force full diagnolization, with all eigenvectors and eigenvalues
@timing
def exactDiagonalization_full(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    eigenValues, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors


# In[ ]:


lan_avg = []
full_avg = []

for i in range(11):  #here put N-1 (maximum)
    N = i+2
    alpha = 1   #density of RBM
    M = alpha*N
    # Constant A
    B = 0.95
    # Variable A
    #B=N/2
    #A = N/2
    #N0 = N/2
    # List of Ak
    Ak = []
    for i in range(N - 1):
        # Constant A
        Ak_i = 1
        # Variable A
        #Ak_i = A / (N0) * np.exp(-i / N0)
        Ak.append(Ak_i)
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,Ak)

    #Exact Diagonalization
    #e, v = exactDiagonalization(ha)
    #Ground state energy
    #edEng = e[0]
    # Ground state
    #edState = v[0]

    #Lists for Histogram Data
    numRuns = 3
    hisIt = np.arange(numRuns)
    runTime_lan = []
    runTime_full = []

    # Cluster multiproccessing
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=3))
    pool = mp.Pool(processes=ncpus)
    # Run Descent
    results_lan = [pool.apply(exactDiagonalization, args=(ha,)) for x in hisIt]
    results_full = [pool.apply(exactDiagonalization_full, args=(ha,)) for x in hisIt]
    
    
    # Get errors for each run in histogram
    for i in range(len(hisIt)):
        #runTime_lan_temp = exactDiagonalization(ha)[1]
        #runTime_full_temp = exactDiagonalization_full(ha)[1]
        runTime_lan_temp = results_lan[i][1]
        runTime_full_temp = results_full[i][1]
        
        runTime_lan.append(runTime_lan_temp)
        runTime_full.append(runTime_full_temp)
        print('runTime_lan', runTime_lan_temp)
        print('runTime_full', runTime_full_temp)

        
    #average the runtime for every choice of N
    lan_avg.append(averageOfList(runTime_lan))
    full_avg.append(averageOfList(runTime_full))
    
    
#Save data to JSON file
data = [lan_avg, full_avg]
fileName = "2021_summer_data/runTime_exact_con.json"
open(fileName, "w").close()
with open(fileName, 'a') as file:
    for item in data:
        line = json.dumps(item)
        file.write(line + '\n')
print('SAVED')


# In[ ]:





# In[ ]:




