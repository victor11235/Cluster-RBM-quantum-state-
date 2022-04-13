#!/usr/bin/env python
# coding: utf-8

# In[7]:
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
import sys
sys.path.append("/home/victor1/penalty_method")
import expect_grad_ex
import vmc_ex
import jax
import pickle
import copy
import optax
# In[8]:


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

def exactDiagonalization_full(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    eigenValues, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err_sta(state, edState,N):
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    waveFunctionErr = 1 - (np.linalg.norm(overlap))**2
    return waveFunctionErr

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
        self.vs = nk.vqs.MCState(self.sampler, self.machine, n_samples=5000, n_discard_per_chain=500) #discarded number of samples 
        #at the beginning of the MC chain

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output, state_list, shift_list):
        self.vs.init_parameters(jax.nn.initializers.normal(stddev=0.25))
        gs = vmc_ex.VMC_ex(hamiltonian=self.hamiltonian, optimizer=self.optimizer, variational_state=self.vs, preconditioner=self.sr, 
                   state_list = state_list, shift_list = shift_list)
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(out=output, n_iter=8000,show_progress = False)
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
        # Outputs the final energy, the final state, and the runtime
        parameters = self.vs.parameters
        # Outputs the final energy, the final state, and the runtime
        return finalEng, state, parameters


def runDescentCS(N,B,Ak,alpha, state_list, shift_list,i, j):
    # reset the seed to avoid seed inheritance for concurrent processes
    np.random.seed()
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam(N,B,Ak)
    # RBM Spin Machine
    ma = nk.models.RBM(alpha=alpha, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma) #an instance of class RBM
    # Run RBM
    eng, state, param = rbm("2021_summer_data/2022_winter_data/excited_Logs_N"+str(N)+"_i"+str(i)+"_j"+str(j), state_list, shift_list)     #where _call_ will be invoked
    #and store the variational parameters 
    #fileName = "2021_sumer_data/2022_winter_data/parameter_N"+str(N)+"i"+str(i)+"j"+str(i)+".json"
    #file = open(fileName,'wb')
    #pickle.dump(param,file)
    return eng, state, param

def runDescentCS_mp(N,B,Ak,alpha, para_list, shift_list, i, j):
    #first create variational state function here then initialize its parameters accordingly
    ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    sampler = nk.sampler.MetropolisLocal(hilbert=hi)
    state_list = []
    for i in range(len(shift_list)):
        
        vs = nk.vqs.MCState(sampler, ma, n_samples=5000, n_discard_per_chain=500)
        vs.init_parameters(jax.nn.initializers.normal(stddev=0.25))
        vs.parameters = para_list[i]
        state_list.append(vs)
    
    #now put everything into runDescentCS
    eng, state, param = runDescentCS(N, B, Ak, alpha, state_list, shift_list, i ,j)
    return eng, state, param
    

N = 6
Ak = []

alpha = 1   #density of RBM
M = alpha*N
# Constant A
B = 0.95
# Variable A
A = N/2
N0 = N/2
for i in range(N-1):
    # Constant A
    #Ak_i = 1
    # Variable A
    Ak_i = A / (N0) * np.exp(-i / N0)
    Ak.append(Ak_i)
    
# Define hamiltonian and hilbert space
ha, hi = CSHam(N,B,Ak)

#Exact Diagonalization
e, v = exactDiagonalization_full(ha)

#prepare the state_list and shift_list for calculating the excited states
ii = 1   #this indicates the level of final state, ex. 0 for ground state
#ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
#sampler = nk.sampler.MetropolisLocal(hilbert=hi)
#vs1 = nk.vqs.MCState(sampler, ma, n_samples=1000, n_discard_per_chain=100)
#vs1.init_parameters(jax.nn.initializers.normal(stddev=0.25))
#para1 = vs1.parameters
fileName = "2021_summer_data/2022_winter_data/post_selection_data/parameter_N6i0j38_best.json"
file = open(fileName,'rb')
a1 = pickle.load(file)

para_list = [a1]
shift_list = [0.3]


#Lists for Histogram Data
numRuns = 50
hisIt = np.arange(numRuns)
state_er_final_list = []
state_final_list = []
eng_final_list = []

# Cluster multiproccessing
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=15))
pool = mp.Pool(processes=ncpus)
# Run Descent
results = []
for j in hisIt:
    result = pool.apply_async(runDescentCS_mp, args=(N,B,Ak,alpha,para_list, shift_list,ii, j)) 
    results.append(result)

pool.close()
pool.join()

resultsSR = [r.get() for r in results]

# Get errors for each run in histogram
for i in range(len(hisIt)):
    #eng_list_temp, state_list_temp = combined(N,B,Ak,alpha)
    eng_list_temp, state_list_temp, param_temp = resultsSR[i]
    eng_final_list.append(eng_list_temp)
    state_final_list.append(state_list_temp)
    er_list_temp = err_sta(state_list_temp, v[ii], N)
    state_er_final_list.append(er_list_temp)
    #and store the variational parameters 
    fileName = "2021_summer_data/2022_winter_data/parameter_N"+str(N)+"i"+str(ii)+"j"+str(i)+".json"
    file = open(fileName,'wb')
    pickle.dump(param_temp,file)
    print('Iteration #', i)
    
#Save data to JSON file
data = [eng_final_list, state_er_final_list]
fileName = "2021_summer_data/2022_winter_data/ex_penalty_N"+str(N)+"_i" + str(ii)+".json"
open(fileName, "w").close()
with open(fileName, 'a') as file:
    for item in data:
        line = json.dumps(item)
        file.write(line + '\n')
print('SAVED')

for i in range(len(hisIt)):
    fileName_state = "2021_summer_data/2022_winter_data/ex_penalty_state_array_N"+str(N)+"_i"+str(ii)+"_j"+str(i)+".json"
    np.savetxt(fileName_state, state_final_list[i])
print('State list saved')

#notes: i means which excited state it is, 0 for ground state etc. j means which run it is.





# In[ ]:




