# TSP QAOA
# ..:$ python3 code_004.py 

import networkx as nx
import math
import numpy as np
from scipy.optimize import minimize
from qxelarator import qxelarator
import matplotlib.pyplot as plt
import re

######################################################

ptrn = re.compile('\(([+-]\d+.*\d*),([+-]\d+.*\d*)\)\s[|]([0-1]*)>') # Regular expression to extract the amplitudes from the string returned by get_state()
isv_prob = True # True if the internal state vector is accessed using get_state() instead of measurement aggregates over multiple shots
shots = 500 # If isv_prob is false, the experiment is run over multple shots for measurement aggregate. Should be some factor of number of qubits to have the same precision

track_opt = []
track_optstep = 0
track_probs = []

class QAOA(object):

    def __init__(self, maxiter):
        self.qx = qxelarator.QX()
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':maxiter, 
                                 'ftol':1.0e-6, 'xtol':1.0e-6, 'disp':True, 'return_all':True}}
        self.p_name = "test_output/qaoa_run.qasm"
        self.expt = 0    
    
    def qaoa_run(self, wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas):
        n_qubits = len(list(wsopp.keys())[0])
        pqasm = []
        coeffs = []
        ang_nos = []
        params = []
        for gate in initstate:
            pqasm.append(gate)
        for p in range(0,steps):
            for gate in ansatz:
                pqasm.append(gate)
            coeffs = np.hstack((coeffs,cfs))
            ang_nos = np.hstack((ang_nos,aid))
            params.append(init_gammas[p])
            params.append(init_betas[p]) 

        def qasmify(params, wpp):
            global isv_prob
            prog = open(self.p_name,"w")
            prog.write("# De-parameterized QAOA ansatz\n")
            prog.write("version 1.0\n")
            prog.write("qubits "+str(n_qubits)+"\n")
            
            prog.write(".qk(1)\n")
            
            # De-parameterize pqasm
            a_id = 0
            a_ctr = 0
            c_ctr = 0
            for i in pqasm:
                # 1-qubit parametric gates
                if i[0] == 'rx' or i[0] == 'ry' or i[0] == 'rz':
                    prog.write(i[0]+" q["+str(i[1])+"],"+str(coeffs[c_ctr]*params[a_id])+"\n")
                    c_ctr += 1
                    a_ctr += 1
                    if a_ctr >= ang_nos[a_id]:
                        a_id += 1
                        a_ctr = 0
                # 1-qubit discrete gates
                elif i[0] == 'x' or i[0] == 'y' or i[0] == 'z' or i[0] == 'h':
                    prog.write(i[0]+" q["+str(i[1])+"]\n")
                # 2-qubit discrete gates
                else:
                    prog.write(i[0]+" q["+str(i[1][0])+"],q["+str(i[1][1])+"]\n")
            
            # Pre-rotation for Z-basis measurement
            tgt = n_qubits-1
            for pt in wpp:
                if pt == "X":
                    prog.write("ry q"+str(tgt)+",1.5708\n")
                elif pt == "Y":
                    prog.write("rx q"+str(tgt)+",-1.5708\n")
                # else Z or Identity
                tgt -= 1

            # Measure all
            if not isv_prob:
                for i in range(n_qubits):
                    prog.write("measure q["+str(i)+"]\n")

            prog.close()        

        def expectation(params):
            E = 0
            self.expt = 0
            xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
            zsgn = [1,-1]
            isgn = [1,-1]
            global track_probs
            track_probs = np.zeros(2**n_qubits)

            for wpp in wsopp:
                qasmify(params,wpp)
                self.qx.set(self.p_name)

                Epp = 0
                p = np.zeros(2**n_qubits)
                c = np.zeros(n_qubits,dtype=bool)
                for i in range(shots):
                    self.qx.execute()
                    # self.qx.execute(1)
                    for i in range(n_qubits):
                        c[i] = self.qx.get_measurement_outcome(i)
                    idx = sum(v<<i for i, v in enumerate(c[::-1]))    
                    p[idx] += 1/shots

                psgn = [1]
                for pt in wpp:
                    if pt == "X":
                        psgn = np.kron(psgn,xsgn)
                    #elif pt == "Y":
                    #    psgn = np.kron(psgn,xsgn) # TBD
                    elif pt == "Z":
                        psgn = np.kron(psgn,zsgn)
                    else: # Identity
                        psgn = np.kron(psgn,isgn)
                for pn in range(2**n_qubits):
                    Epp += psgn[pn]*p[pn]                
                E += wsopp[wpp]*Epp
                self.expt += E

                if wpp == "I"*n_qubits:
                    track_probs = p

            return E

        def expectation_isv(params):
            global ptrn
            E = 0
            self.expt = 0
            xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
            zsgn = [1,-1]
            isgn = [1,-1]
            global track_probs
            track_probs = np.zeros(2**n_qubits)

            for wpp in wsopp:
                qasmify(params,wpp)
                self.qx.set(self.p_name)

                Epp = 0
                p = np.zeros(2**n_qubits)
                self.qx.execute() 
                isv_str = self.qx.get_state()
                isv = re.findall(ptrn,isv_str)
                for basis in iter(isv):
                    p[int(basis[2],2)] = float(basis[0])**2 + float(basis[1])**2 # Probability is square of modulus of complex amplitude
                
                psgn = [1]
                for pt in wpp:
                    if pt == "X":
                        psgn = np.kron(psgn,xsgn)
                    #elif pt == "Y":
                    #    psgn = np.kron(psgn,xsgn) # TBD
                    elif pt == "Z":
                        psgn = np.kron(psgn,zsgn)
                    else: # Identity
                        psgn = np.kron(psgn,isgn)
                for pn in range(2**n_qubits):
                    Epp += psgn[pn]*p[pn]                
                E += wsopp[wpp]*Epp
                self.expt += E

                if wpp == "I"*n_qubits:
                    track_probs = p
               
            return E

        def intermediate(cb):
            global track_opt
            global track_optstep
            global track_probs
            print("Step: ",track_optstep)
            # print("Current Optimal Parameters: ",cb)
            # print("Current Expectation Value: ",self.expt)
            # print("Current Optimal Probabilities: ",track_probs)
            track_optstep += 1
            # input("Press Enter to continue to step "+str(track_optstep))
            track_opt.append([track_optstep, cb, track_probs])
               
        args = [expectation_isv, params]
        r = self.minimizer(*args, callback=intermediate, **self.minimizer_kwargs) 
        return r

######################################################

"""
Defines the city graph for TSP
"""

# 4 cities undirected complete-graph in unit square lattice

#   0----1
#   |\  /|
#   | \/ |
#   | /\ |
#   |/  \|
#   3----2

# def graph_problem():
#     g = nx.Graph()
#     g.add_edge(0,1,weight=1)
#     g.add_edge(0,2,weight=math.sqrt(2))
#     g.add_edge(0,3,weight=1)
#     g.add_edge(1,2,weight=1)
#     g.add_edge(1,3,weight=math.sqrt(2))
#     g.add_edge(2,3,weight=1)
#     return g

# 3 cities unit distance undirected complete-graph

#       0
#      / \
#     /   \
#    /     \
#   2-------1

def graph_problem():
    g = nx.Graph()
    g.add_edge(0,1,weight=1)
    g.add_edge(0,2,weight=1)
    g.add_edge(1,2,weight=1)
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
    
    penalty = 1e5 # How to set this?
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

                sopp = Iall[:(i*n_cities+p)]+"Z"+Iall[(i*n_cities+p)+1:]
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-dist/4)
                else:
                    wsopp[sopp] = (-dist/4)

                sopp = Iall[:(j*n_cities+q)]+"Z"+Iall[(j*n_cities+q)+1:]
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (-dist/4)
                else:
                    wsopp[sopp] = (-dist/4)

                sopp = sopp[:(i*n_cities+p)]+"Z"+sopp[(i*n_cities+p)+1:] # Both ip,jq
                if sopp in wsopp:
                    wsopp[sopp] = wsopp[sopp] + (dist/4)
                else:
                    wsopp[sopp] = (dist/4)
       
    return shift, wsopp

shift, wsopp = graph_to_wsopp_tsp(g)

# print(shift,len(wsopp))
# for i in wsopp:
#     print(i,wsopp[i]) 

######################################################

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

PRINT("HELLO")
qaoa_obj = QAOA(maxiter)
# res = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# print(res.status, res.fun, res.x)
print(QAOA.track_opt[-1])
# print(sum(track_opt[0][2]))
# # %matplotlib inline
# plt.ylim((0,1))
# plt.plot(track_opt[0][2],'--') # Initial
# plt.plot(track_opt[-1][2]) # Final
# plt.show()



######################################################

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


























######################################################

# from scipy import sparse

# def paulis_to_matrix(pl):
#     """
#     Convert paulis to matrix, and save it in internal property directly.
#     If all paulis are Z or I (identity), convert to dia_matrix.
#     """
#     p = pl[0]
#     hamiltonian = p[0] * to_spmatrix(p[1])
#     for idx in range(1, len(pl)):
#         p = pl[idx]
#         hamiltonian += p[0] * to_spmatrix(p[1])
#     return hamiltonian

# def to_spmatrix(p):
#     """
#     Convert Pauli to a sparse matrix representation (CSR format).
#     Order is q_{n-1} .... q_0, i.e., $P_{n-1} \otimes ... P_0$
#     Returns:
#         scipy.sparse.csr_matrix: a sparse matrix with CSR format that
#         represnets the pauli.
#     """
#     mat = sparse.coo_matrix(1)
#     for z in p:
#         if not z:  # I
#             mat = sparse.bmat([[mat, None], [None, mat]], format='coo')
#         else:  # Z
#             mat = sparse.bmat([[mat, None], [None, -mat]], format='coo')
#     return mat.tocsr()

######################################################