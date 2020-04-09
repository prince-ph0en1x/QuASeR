from networkx import networkx as nx
import math
import numpy as np
# import matplotlib.pyplot as plt
######################################################

"""
Defines the city graph for TSP: DNA example in paper
"""
def graph_problem():
    g = nx.DiGraph()
    g.add_edge(0,1,weight=-7)
    g.add_edge(1,2,weight=-7)
    g.add_edge(2,3,weight=-7)
    g.add_edge(3,0,weight=-9)
    g.add_edge(1,0,weight=-3)
    g.add_edge(2,1,weight=-3)
    g.add_edge(3,2,weight=-3)
    g.add_edge(0,3,weight=-1)
    g.add_edge(0,2,weight=-4)
    g.add_edge(1,3,weight=-4)
    g.add_edge(2,0,weight=-6)
    g.add_edge(3,1,weight=-6)
    return g
g = graph_problem()

######################################################

"""
Converts the TSP graph to wsopp for problem Hamiltonian for QUBO/Ising model
"""

def graph_to_wsopp_tsp(g):
    # for i in g.edges():
    #     print(i,g.edges[i]['weight'])
    wsopp = {}
    n_cities = len(g.nodes())
    n_timeslots = n_cities
    n_qubits = n_cities*n_timeslots # MSQ = C0T0, LSQ = C3T3
    Iall = "I"*n_qubits
    
    penalty = 1e1 # How to set this?
    shift = 2*penalty*n_cities   # What is this?
    
    # penalty: CiTp --> CiTp
    for i in range(n_cities):
        for p in range(n_timeslots):
            # zp = np.zeros(n_qubits, dtype=np.bool)
            # zp[i * n_cities + p] = True
            # print(zp)
            shift += -penalty

            sopp = Iall[:(i*n_cities+p)]+"Z"+Iall[(i*n_cities+p)+1:]
            if sopp in wsopp:
                wsopp[sopp] = wsopp[sopp] + penalty
            else:
                wsopp[sopp] = penalty

    # -0.5*penalty: CiTp --> CiTp
    # -0.5*penalty: CjTp --> CjTp
    # +0.5*penalty: CiTp --> CjTp
    for p in range(n_timeslots):
        for i in range(n_cities):
            for j in range(i):
                shift += penalty / 2

                
                sopp = Iall[:(i*n_cities+p)]+"Z"+Iall[(i*n_cities+p)+1:]
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-penalty/2)
                else:
                    wsopp[sopp] = (-penalty/2)

                sopp = Iall[:(j*n_cities+p)]+"Z"+Iall[(j*n_cities+p)+1:]
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-penalty/2)
                else:
                    wsopp[sopp] = (-penalty/2)

                sopp = sopp[:(i*n_cities+p)]+"Z"+sopp[(i*n_cities+p)+1:] # Both i,j
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (penalty/2)
                else:
                    wsopp[sopp] = (penalty/2)    
              
    # -0.5*penalty: CiTp --> CiTp
    # -0.5*penalty: CiTq --> CiTq
    # +0.5*penalty: CiTp --> CiTq
    for i in range(n_cities):
        for p in range(n_timeslots):
            for q in range(p):
                shift += penalty / 2

                sopp = Iall[:(i*n_cities+p)]+"Z"+Iall[(i*n_cities+p)+1:]
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-penalty/2)
                else:
                    wsopp[sopp] = (-penalty/2)

                sopp = Iall[:(i*n_cities+q)]+"Z"+Iall[(i*n_cities+q)+1:]
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-penalty/2)
                else:
                    wsopp[sopp] = (-penalty/2)

                sopp = sopp[:(i*n_cities+p)]+"Z"+sopp[(i*n_cities+p)+1:] # Both p,q
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (penalty/2)
                else:
                    wsopp[sopp] = (penalty/2)

    # -0.25*dist(i,j): CiTp --> CiTp
    # -0.25*dist(i,j): CjTq --> CjTq
    # +0.25*dist(i,j): CiTp --> CjTq
    for i in range(n_cities):
        for j in range(n_cities):
            if i == j:
                continue
            for p in range(n_timeslots):
                q = (p + 1) % n_timeslots
                dist = g.get_edge_data(i,j)['weight']
                shift += dist / 4
                # print(i,j,p,q)

                sopp = Iall[:(i*n_cities+p)]+"Z"+Iall[(i*n_cities+p)+1:]
                # print(sopp,-dist/4)
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-dist/4)
                else:
                    wsopp[sopp] = (-dist/4)

                sopp = Iall[:(j*n_cities+q)]+"Z"+Iall[(j*n_cities+q)+1:]
                # print(sopp,-dist/4)
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-dist/4)
                else:
                    wsopp[sopp] = (-dist/4)

                sopp = sopp[:(i*n_cities+p)]+"Z"+sopp[(i*n_cities+p)+1:] # Both ip,jq
                # print(sopp,dist/4)
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (dist/4)
                else:
                    wsopp[sopp] = (dist/4) 
       
    return shift, wsopp

shift, wsopp = graph_to_wsopp_tsp(g)

# print(shift,len(wsopp))
# for i in wsopp:
#     print(i,wsopp[i]) 

#####################################################

initstate = []
for i in range(0,len(g.nodes())): # Reference state preparation
    initstate.append(("h",i))

# Refer: Rigetti --> Forest --> PyQuil --> paulis.py --> exponential_map()
def ansatz_pqasm_tsp(wsopp):
    ansatz = [] # qasm tokens
    coeffs = [] # Weights for the angle parameter for each gate
    angles = [0,0] # Counts for [cost,mixing] Hamiltonian angles
    
    for i in wsopp: # Find better way to find length of one key in a dict
    	n_qubits = len(i)
    	break
    
    # Cost Hamiltonian # CHECK THIS PART WITH RIGETTI AND TSP FORM

    for i in wsopp:

    	if i.count('Z') == 1:
    		ansatz.append(("rz",i.find('Z')))
    		coeffs.append(+2*+1)
    		angles[0] += 1 # gamma

    	else:
    		cq = i.find('Z')
    		tq = (cq+1)+i[(cq+1):].find('Z')
    		ansatz.append(("cnot",[cq,tq]))
    		ansatz.append(("rz",tq))
    		coeffs.append(+2*+1)
    		angles[0] += 1 # gamma
    		ansatz.append(("cnot",[cq,tq]))  
    		# print(i,wsopp[i],cq,tq)

    # Mixing Hamiltonian
    
    # +I_all (doesn't matter which qubit)
    ansatz.append(("x",0))
    ansatz.append(("rz",0)) # Phase(coeff) = RZ(coeff) * exp(i*coeff/2)
    coeffs.append(-1*+1)
    angles[1] += 1 # beta 
    ansatz.append(("x",0))
    ansatz.append(("rz",0)) # Phase(coeff) = RZ(coeff) * exp(i*coeff/2)
    coeffs.append(-1*+1)
    angles[1] += 1 # beta
    
    # -X_i
    for i in range(n_qubits):
        ansatz.append(("h",i))
        ansatz.append(("rz",i))
        coeffs.append(+2*-1)
        angles[1] += 1 # beta
        ansatz.append(("h",i))

    return ansatz, coeffs, angles

ansatz, cfs, aid = ansatz_pqasm_tsp(wsopp)
# print(ansatz)

steps = 1 # Number of steps (QAOA blocks per iteration)

# Initial angle parameters for Hamiltonians cost (gammas) and mixing/driving (betas)
init_gammas = np.random.uniform(0, 2*np.pi, steps) 
init_betas = np.random.uniform(0, 2*np.pi, steps)

######################################################

maxiter = 1

from QAOA import QAOA, track_opt
print(init_gammas, init_betas)
qaoa_obj = QAOA(maxiter)
for i in range(0,len(wsopp)):
    print("#",end="")
print()#, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
res = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# print(res.status, res.fun, res.x)
# res = qaoa_obj.qaoa_test(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# print(res)
print(track_opt[-1])
# # print(track_opt[-1])
# # print(sum(track_opt[0][2]))
# # # %matplotlib inline
# # plt.ylim((0,1))
# # plt.plot(track_opt[0][2],'--') # Initial
# # plt.plot(track_opt[-1][2]) # Final
# # plt.show()



# ######################################################

# from pyquil.paulis import *
# # from pyquil.gates import *
# ps = sI(2)
# ps = 0.2*(sI(0) - sZ(1)*sZ(2))	
# # for i in range(0,4):
# # 	ps -= sX(i)
# print(ps)

# for pt in ps: # pauli term in pauli sum
# 	print("\nPauli Term: ",pt)
# 	empt = exponential_map(pt)
# 	for gs in empt(1.222):
# 		print(gs)

######################################################

# def graph_to_pqasm(g,n_qubits):
#     coeffs = [] # Weights for the angle parameter for each gate
#     angles = [0,0] # Counts for [cost,mixing] Hamiltonian angles
#     Iall = ""
#     for i in range(n_qubits):
#         Iall += "I"
#     ansatz = [] # qasm tokens
#     for i,j in g.edges():
#         # 0.5*Z_i*Z_j
#         ansatz.append(("cnot",[i,j]))
#         ansatz.append(("rz",j))
#         coeffs.append(2*0.5)
#         angles[0] += 1 # gamma: cost Hamiltonian
#         ansatz.append(("cnot",[i,j]))
#         # -0.5*I_0
#         ansatz.append(("x",0))
#         ansatz.append(("rz",0))
#         coeffs.append(-1*-0.5)
#         angles[0] += 1 # gamma: cost Hamiltonian
#         ansatz.append(("x",0))
#         ansatz.append(("rz",0))
#         coeffs.append(-1*-0.5)
#         angles[0] += 1 # gamma: cost Hamiltonian
#     for i in g.nodes():
#         # -X_i
#         ansatz.append(("h",i))
#         ansatz.append(("rz",i))
#         coeffs.append(2*-1)
#         angles[1] += 1 # beta: mixing Hamiltonian
#         ansatz.append(("h",i))
#     return ansatz, coeffs, angles

# ansatz, cfs, aid = graph_to_pqasm(g,len(g.nodes()))

# steps = 4 # number of steps (QAOA blocks per iteration)

# Initial angle parameters for Hamiltonians cost (gammas) and mixing/driving (betas)

# init_gammas = np.random.uniform(0, 2*np.pi, steps) 
# init_betas = np.random.uniform(0, np.pi, steps)

# init_gammas = [0, 0]
# init_betas = [0, 0]

# Optimization terminated successfully.
#          Current function value: 0.000000
#          Iterations: 19
#          Function evaluations: 189
# 0 0.0 [0.76556019 0.65266102 2.31622719 0.24012393 1.19432261 0.70770831
#  2.87653068 2.75631259]
# [18, array([0.76556019, 0.65266102, 2.31622719, 0.24012393, 1.19432261,
#        0.70770831, 2.87653068, 2.75631259]), array([2.35625479e-02, 4.96280102e-02, 4.91347731e-02, 1.26990289e-02,
#        4.59621422e-05, 2.46748704e-02, 3.01462133e-02, 3.01462133e-02,
#        4.59621422e-05, 2.46748704e-02, 7.09195465e-02, 7.09195465e-02,
#        4.67071649e-03, 4.68969246e-02, 1.26990289e-02, 4.91347731e-02,
#        4.91347731e-02, 1.26990289e-02, 4.68969246e-02, 4.67071649e-03,
#        7.09195465e-02, 7.09195465e-02, 2.46748704e-02, 4.59621422e-05,
#        3.01462133e-02, 3.01462133e-02, 2.46748704e-02, 4.59621422e-05,
#        1.26990289e-02, 4.91347731e-02, 4.96280102e-02, 2.35625479e-02])]

######################################################

# maxiter = 20

# qaoa_obj = QAOA(maxiter, shots)
# res = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# print(res.status, res.fun, res.x)
# print(track_opt[-1])
# print(sum(track_opt[0][2]))
# # %matplotlib inline
# plt.ylim((0,1))
# plt.plot(track_opt[0][2],'--') # Initial
# plt.plot(track_opt[-1][2]) # Final
# plt.show()

######################################################























