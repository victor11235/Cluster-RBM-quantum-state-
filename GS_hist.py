#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
plt.style.use('seaborn')
from scipy.stats import norm


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


def exactDiagonalization(hamiltonian):
    # Changes Hamiltonian to matrix form, where hamiltonian of interest is sparse in matrix form
    #haMatrix = hamiltonian.to_sparse()
    # Gets eigenvalues and vectors, where the built-in function uses 
    eigenValues, v = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)

    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors


# NetKet RBM with stochastic reconfiguration descent
class RBM:
    def __init__(self, N, hamiltonian, hilbertSpace, machine):
        # Assign inputsv[:, i]
        self.hamiltonian, self.hilbertSpace, self.machine, self.N = hamiltonian, hilbertSpace, machine, N
        # Define sampler
        self.sampler = nk.sampler.MetropolisLocal(hilbert=hilbertSpace)
        # Define optimizer
        self.optimizer = nk.optimizer.Sgd(learning_rate=0.02)
        # Define Stochastic reconfiguration
        self.sr = nk.optimizer.SR(diag_shift=0.01) #diagnol shift, its role as regularizer? seems to take a different form as 
        #compared to the version I have seen
        # Variational state
        self.vs = nk.variational.MCState(self.sampler, self.machine, n_samples=1000, n_discard=100) #discarded number of samples 
        #at the beginning of the MC chain

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        self.vs.init_parameters(nk.nn.initializers.normal(stddev=0.25))
        gs = nk.VMC(hamiltonian=self.hamiltonian, optimizer=self.optimizer, variational_state=self.vs, sr=self.sr)
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(n_iter=1000, out=output, show_progress = False) #callback = [nk.callbacks.EarlyStopping(monitor='mean', patience=80,                   min_delta=0.00001), nk.callbacks.Timeout(600)])
        
        end = time.time()
        runTime = end - start
        # Import the data from log file
        data = json.load(open(output + '.log'))
        # Extract the relevant information
        # iters = data["Energy"]["iters"]
        energy_RBM = data["Energy"]["Mean"]["real"] #get the real part of the mean energy
       
        # finalEng = energy_RBM[-1]
        finalEng = reduce(lambda x, y: x if y is None else y, energy_RBM)
        # Get machine statethe state of the machine as an array
        state = self.vs.to_array()
        # Get the total number of iterations
        #iters = data["Energy"]["iters"]
        #n_iters = reduce(lambda x, y: x if y is None else y, iters) + 1
        # Outputs the final energy, the final state, the runtime, and the number of iterations
        return finalEng, state, runTime
    
    
# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng,N):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    waveFunctionErr = 1 - (np.linalg.norm(overlap))**2
    return engErr, waveFunctionErr


# Combines all steps into a function to run on the cluster
def runDescentCS(N,B,Ak,alpha,j):
    # reset the seed to avoid seed inheritance for concurrent processes
    np.random.seed()
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam(N,B,Ak)
    # RBM Spin Machine
    ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma) #an instance of class RBM
    # Run RBM
    eng, state, runTime=  rbm("2021_summer_data/2022_winter_data/GS_hist"+str(N)+str(j)) #where _call_ will be invoked
    return eng, state, runTime


for i in range(11):  #here put N-1 (maximum)
    N = i+2
    alpha = 1   #density of RBM
    M = alpha*N
    # Constant A
    B = 0.95
    # Variable A
    #B=N/2
    A = N/2
    N0 = N/2
    # List of Ak
    Ak = []
    for i in range(N - 1):
        # Constant A
        #Ak_i = 1
        # Variable A
        Ak_i = A / (N0) * np.exp(-i / N0)
        Ak.append(Ak_i)
    # Define hamiltonian and hilbert space
    ha, hi = CSHam(N,B,Ak)

    #Exact Diagonalization
    e, v = exactDiagonalization(ha)
    #Ground state energy
    edEng = e[0]
    # Ground state
    edState = v[0]

    #Lists for Histogram Data
    numRuns = 50
    hisIt = np.arange(numRuns)
    engErr = []
    stateErr = []
    runTime = []

    
    # Cluster multiproccessing
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=25))
    pool = mp.Pool(processes=ncpus)
    # Run Descent
    results = []
    for j in hisIt:
        result = pool.apply_async(runDescentCS, args=(N,B,Ak,alpha,j)) 
        results.append(result)
    
    pool.close()
    pool.join()
    resultsSR = [r.get() for r in results]
    
    
    
    # Get errors for each run in histogram
    for i in range(len(hisIt)):
        #engTemp, stateTemp, runTimeTemp = runDescentCS(N,B,Ak,alpha)
        engTemp, stateTemp, runTimeTemp = resultsSR[i]
        runTime.append(runTimeTemp)
        #n_iters.append(n_itersTemp)
        errSR = err(np.asmatrix(stateTemp), edState, engTemp, edEng,N) #make state vector as matrix data-type
        engErr.append(errSR[0])
        stateErr.append(errSR[1])
        #print('n_iters', n_iters)
        #print('Eng error ', engErr)
        #print('State error ', stateErr)

        
    #Save data to JSON file
    data = [engErr, stateErr, runTime]
    fileName = "2021_summer_data/2022_winter_data/sum_GS_hist"+str(N)+"M" + str(M)+".json"
    open(fileName, "w").close()
    with open(fileName, 'a') as file:
        for item in data:
            line = json.dumps(item)
            file.write(line + '\n')
    print('SAVED')

