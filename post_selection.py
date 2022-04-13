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
import copy
import pickle
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

def CSHam_ex(N, B, Ak, state_list, Filling):
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
    # now we add the extra gap filler
    fill_Op = np.zeros((2**N, 2**N), dtype = 'complex128')
    for j in range(len(Filling)):
        fill_Op += Filling[j] * np.outer(state_list[j], np.conj(state_list[j])) 
    
    operators.append(fill_Op)
    sites.append(np.arange(0,N).tolist())
    # Create hamiltonian
    hamiltonian = nk.operator.LocalOperator(hilbertSpace, operators=operators, acting_on=sites, dtype='complex128')
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
        self.optimizer = optax.rmsprop(learning_rate = 0.01, decay = 0.95, eps = 10**-2, initial_scale = 0.01)
        # Define Stochastic reconfiguration
        self.sr = nk.optimizer.SR(diag_shift=0.0001) #diagnol shift, its role as regularizer? seems to take a different form as 
        #compared to the version I have seen
        # Variational state
        self.vs = nk.variational.MCState(self.sampler, self.machine, n_samples=20000, n_discard=2000) #discarded number of samples 
        #at the beginning of the MC chain

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        self.vs.init_parameters(nk.nn.initializers.normal(stddev=0.25))
        gs = nk.VMC(hamiltonian=self.hamiltonian, optimizer=self.optimizer, variational_state=self.vs, precondioner=self.sr)

        # Set the output files as well as number of iterations in the descent
        gs.run(n_iter=10000, out=output, show_progress = False)
        

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
        return finalEng, state


def runDescentCS_ex(N,B,Ak,alpha,state_list, Filling, i, j):
    np.random.seed()
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam_ex(N,B,Ak,state_list, Filling)
    # RBM Spin Machine
    ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma) #an instance of class RBM
    # Run RBM
    eng, state = rbm("2021_summer_data/excited_hist/extra_ex_data_var_post_5_excited_Logs"+str(N) + str(i)+'_' + str(j)) #where _call_ will be invoked
    return eng, state
 


    
def combined(N,B,Ak,alpha,j):

    eng = []
    state = []
    
    eng = np.loadtxt("2021_summer_data/temp_data/eng_4.json")
    state = np.loadtxt("2021_summer_data/temp_data/state_4.json", dtype = complex)
    eng = eng.tolist()
    state = state.tolist()
    
    
    v_0, v_1, v_2, v_3, v_4 = state[0], state[1], state[2], state[3], state[4]
    e_0, e_1, e_2, e_3, e_4 = eng[0], eng[1], eng[2], eng[3], eng[4]
    
    zeros = np.zeros(2**N)
    fill = 2 * np.abs(e_0)
    
    state_list_5 = [v_0, v_1, v_2, v_3, v_4]
    Filling_5 = [fill, fill, fill, fill, fill]
    e_5, v_5 = runDescentCS_ex(N, B, Ak, alpha, state_list_5, Filling_5, 5, j)
    
    return e_5, v_5

        


# In[ ]:


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


#Lists for Histogram Data
numRuns = 20
hisIt = np.arange(numRuns)

state_er_final_list = []
state_final_list = []
eng_final_list = []
eng_er_final_list = []

# Cluster multiproccessing
ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=20))
pool = mp.Pool(processes=ncpus)
# Run Descent
results = []
for j in hisIt:
    result = pool.apply_async(combined, args=(N,B,Ak,alpha, j)) 
    results.append(result)

pool.close()
pool.join()
resultsSR = [r.get() for r in results]

# Get errors for each run in histogram
for i in range(len(hisIt)):
    #eng_list_temp, state_list_temp = combined(N,B,Ak,alpha)
    eng_temp, state_temp = resultsSR[i]
    eng_final_list.append(eng_temp)
    state_final_list.append(state_temp)
    
    er_state_temp = err_sta(state_temp, v[5], N)
    er_eng_temp = eng_temp - e[5]
    
    eng_er_final_list.append(er_eng_temp)
    state_er_final_list.append(er_state_temp)
    print('Iteration #', i)
    
#Save data to JSON file
data = [eng_final_list, eng_er_final_list, state_er_final_list]
fileName = "2021_summer_data/extra_ex_hist_var_post_5_excited_N"+str(N)+"M" + str(M)+".json"
open(fileName, "w").close()
with open(fileName, 'a') as file:
    for item in data:
        line = json.dumps(item)
        file.write(line + '\n')
print('SAVED')


fileName_state = "2021_summer_data/State_list/extra_var_post_5_excited_states.json"
np.savetxt(fileName_state, state_final_list)
print('State list saved')

# In[ ]:





# In[ ]:




