#!/usr/bin/env python
# coding: utf-8

# In[24]:


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


# In[25]:


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

def CSHam_ex(N, B, Ak, GS, Filling):
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
    #GS_s = scipy.sparse.csr_matrix(GS)
    #fill_Op = Filling * scipy.sparse.kron(GS_s, scipy.sparse.spmatrix.getH(GS_s))
    fill_Op = Filling * np.outer(GS, np.conj(GS))
    operators.append(fill_Op)
    sites.append(np.arange(0,N).tolist())
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

def exactDiagonalization_full(hamiltonian):
    # Changes Hamiltonian to matrix form
    haMatrix = hamiltonian.to_dense()
    # Gets eigenvalues and vectors
    eigenValues, v = np.linalg.eigh(haMatrix)
    # Orders from smallest to largest
    eigenVectors = [v[:, i] for i in range(len(eigenValues))]
    return eigenValues, eigenVectors

# Error Calculation (Input: the found state, the state from exact diagonalization, the found energy, the energy from exact diagonalization)
def err(state, edState, eng, edEng,N):
    engErr = np.abs(eng - edEng)
    overlap = np.dot(state.conj().reshape(2**N, 1).T, edState.reshape(2**N, 1))
    waveFunctionErr = 1 - (np.linalg.norm(overlap))**2
    return engErr, waveFunctionErr


# In[26]:


# NetKet RBM with stochastic reconfiguration descent
class RBM:
    def __init__(self, N, hamiltonian, hilbertSpace, machine):
        # Assign inputsv[:, i]
        self.hamiltonian, self.hilbertSpace, self.machine, self.N = hamiltonian, hilbertSpace, machine, N
        # Define sampler
        self.sampler = nk.sampler.MetropolisLocal(hilbert=hilbertSpace)
        # Define optimizer
        self.optimizer = nk.optimizer.Sgd(learning_rate=0.01)
        # Define Stochastic reconfiguration
        self.sr = nk.optimizer.SR(diag_shift=0.01) #diagnol shift, its role as regularizer? seems to take a different form as 
        #compared to the version I have seen
        # Variational state
        self.vs = nk.variational.MCState(self.sampler, self.machine, n_samples=1000, n_discard=100) #discarded number of samples 
        #at the beginning of the MC chain

    # Output is the name of the output file in which the descent data is stored
    def __call__(self, output):
        self.vs.init_parameters(nk.nn.initializers.normal(stddev=0.25))
        gs = nk.VMC(hamiltonian=self.hamiltonian, optimizer=self.optimizer, variational_state=self.vs, precondioner=self.sr)
        # Start timing
        start = time.time()
        # Set the output files as well as number of iterations in the descent
        gs.run(out=output, n_iter=1000, show_progress = False)
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
        return finalEng, state, runTime
    
# Combines all steps into a function to run on the cluster
def runDescentCS_ex(N,B,Ak,alpha,GS,Filling, i):
    np.random.seed()
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam_ex(N,B,Ak,GS,Filling)
    # RBM Spin Machine
    ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma) #an instance of class RBM
    # Run RBM
    eng, state, runTime = rbm("2021_summer_data/excited/ex_Logs"+str(N)+"run"+str(i)+"Filling"+str(round(Filling,2))) #where _call_ will be     invoked
    return eng, state, runTime

# Combines all steps into a function to run on the cluster
def runDescentCS(N,B,Ak,alpha):
    np.random.seed()
    # Define hamiltonian and hibert space (need to do this here cause can't use netket objects as input to use multiprocessing functions)
    ha, hi = CSHam(N,B,Ak)
    # RBM Spin Machine
    ma = nk.models.RBM(alpha=1, dtype=complex,use_visible_bias=True, use_hidden_bias=True)
    # Initialize RBM
    rbm = RBM(N, ha, hi, ma) #an instance of class RBM
    # Run RBM
    eng, state, runTime = rbm("2021_summer_data/excited/Logs"+str(N)) #where _call_ will be invoked
    return eng, state, runTime


# Here we find the different values of gap filling for testing
Ak = []
N = 5
alpha = 1
M = alpha*N

B=0.95
# Variable A
#A = N/2
#N0 = N/2
for i in range(4):
        # Constant A
    Ak_i = 1
        # Variable A
        #Ak_i = A / (N0) * np.exp(-i / N0)
    Ak.append(Ak_i)

ha, hi = CSHam(N, B, Ak)

e,v  = exactDiagonalization_full(ha)
print('exact ground state is',e)
fill_1 = np.abs(e[1]-e[0])
fill_2 = fill_1 + 0.5 * np.abs(e[2]-e[1])
fill_3 = fill_1 + np.abs(e[2]-e[1])
fill_4 = np.abs(e[0]) + np.abs(e[-1]) + 1

filling = [fill_1, fill_2, fill_3, fill_4]
print(filling)


#Ground state energy
edEng = e[0]
# Ground state
edState = v[0]

#Lists for Histogram Data
numRuns = 1
hisIt = np.arange(numRuns)
engErr = []
stateErr = []
runTime = []
GS = []

# Get errors for each run in histogram
for i in range(len(hisIt)):
    engTemp, stateTemp, runTimeTemp = runDescentCS(N,B,Ak,alpha)
    #engTemp, stateTemp, runTimeTemp = resultsSR[i]
    runTime.append(runTimeTemp)
    GS.append(stateTemp)
    errSR = err(np.asmatrix(stateTemp), edState, engTemp, edEng,N) #make state vector as matrix data-type
    engErr.append(errSR[0])
    stateErr.append(errSR[1])
    print('Eng error ', engErr)
    print('State error ', stateErr)
print(GS)



#Ground state energy
edEng_ex = e[1]
# Ground state
edState_ex = v[1]



for f_i in range(len(filling)):  
    
    ii = f_i
    
    
    #Lists for Histogram Data
    numRuns_ex = 50
    hisIt_ex = np.arange(numRuns_ex)
    engErr_ex = []
    stateErr_ex = []
    runTime_ex = []
    
     # Cluster multiproccessing
    ncpus = int(os.environ.get('SLURM_CPUS_PER_TASK',default=20))
    pool = mp.Pool(processes=ncpus)
    # Run Descent
    results = []
    for j in hisIt_ex:
        result = pool.apply_async(runDescentCS_ex, args=(N,B,Ak,alpha, GS[0], filling[ii], j)) 
        results.append(result)
    
    pool.close()
    pool.join()
    resultsSR = [r.get() for r in results]
    
    # Get errors for each run in histogram
    for i in range(len(hisIt_ex)):
        #engTemp, stateTemp, runTimeTemp = runDescentCS_ex(N,B,Ak,alpha, GS[0], filling[f_i],i)
        engTemp, stateTemp, runTimeTemp = resultsSR[i]
        runTime_ex.append(runTimeTemp)
        errSR = err(np.asmatrix(stateTemp), edState_ex, engTemp, edEng_ex,N) #make state vector as matrix data-type
        engErr_ex.append(errSR[0])
        stateErr_ex.append(errSR[1])
        print('Eng error ', engErr_ex)
        print('State error ', stateErr_ex)

    #Save data to JSON file
    data = [engErr_ex, stateErr_ex, runTime_ex]
    fileName = "2021_summer_data/Systematic_filling/con_N"+str(N)+"M" + str(M)+"fill"+str(f_i)+".json"
    open(fileName, "w").close()
    with open(fileName, 'a') as file:
        for item in data:
            line = json.dumps(item)
            file.write(line + '\n')
    print('SAVED')


# In[36]:


#take half of the energy gap
avg_eng_er = []
avg_sta_er = []
eng_con = []
    

for f_i in range(4):
    with open("2021_summer_data/Systematic_filling/con_N"+str(N)+"M" + str(M)+"fill"+str(f_i)+".json") as f:
        for line in f:
            eng_con.append(json.loads(line))
    avg_eng_er_temp = np.average(eng_con[3*(f_i)])
    avg_eng_er.append(avg_eng_er_temp)
    avg_sta_er_temp = np.average(eng_con[3*(f_i)+1])
    avg_sta_er.append(avg_sta_er_temp)
print(avg_eng_er)


print(avg_sta_er)


# In[ ]:




