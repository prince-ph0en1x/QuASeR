# TSP QAOA
# ..:$ python3 code_005.py 

import networkx as nx
import math
import numpy as np
from scipy.optimize import minimize
from qxelarator import qxelarator
import matplotlib.pyplot as plt
import re

####################################################################################################
###################################################################################################
###################################################################################################    QAOA
###################################################################################################
###################################################################################################

# If positive, measurement aggregate over multiple shots is taken instead of accessing the internal state vector using get_state()
# Should be some factor of number of qubits to have the same precision for different problem sizes
shots = 0 

# Database to save optimization result from each step
# [step_number, parameters, probabilities, expectation_value]
track_opt = []

# Regular expression to extract the amplitudes from the string returned by get_state()
ptrn = re.compile('\(([+-]\d+.*\d*),([+-]\d+.*\d*)\)\s[|]([0-1]*)>') 

class QAOA(object):

    def __init__(self, maxiter):
        
        self.qx = qxelarator.QX()
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':maxiter, 
                                 'ftol':1.0e-6, 'xtol':1.0e-6, 'disp':True, 'return_all':True}}
        # self.minimizer_kwargs = {'method':'Powell', 'options':{'maxiter':maxiter, 
        #                          'ftol':1.0e-6, 'xtol':1.0e-6, 'disp':True}}
        self.p_name = "test_output/qaoa_run.qasm"
        self.expt = 0    
        self.probs = []
        self.optstep = 0
    
    def qaoa_run(self, wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas):
        
        global shots
        n_qubits = len(list(wsopp.keys())[0])

        # Form Parameterized QASM
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

        '''
        De-parameterizes the variational circuit to cQASM
        Adds pre-rotation based on which Pauli Sum to measure the expectation
        Adds measurments based on type of run
        '''
        def qasmify(params, wpp):

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
            if shots > 0:
                for i in range(n_qubits):
                    prog.write("measure q["+str(i)+"]\n")

            prog.close()        

        '''
        Access the internal state vector of QX to find the aggregate expectation value of each Pauli Product term in the Hamiltonian
        '''
        def expectation_isv(params):
            
            global ptrn
            E = 0
            self.expt = 0
            xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
            zsgn = [1,-1]
            isgn = [1,-1]

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
                    self.probs = p
               
            return E

        def expectation_qst(params):
            E = 0
            # self.expt = 0
            # xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
            # zsgn = [1,-1]
            # isgn = [1,-1]
            # global track_probs
            # track_probs = np.zeros(2**n_qubits)

            # for wpp in wsopp:
            #     qasmify(params,wpp)
            #     self.qx.set(self.p_name)

            #     Epp = 0
            #     p = np.zeros(2**n_qubits)
            #     c = np.zeros(n_qubits,dtype=bool)
            #     for i in range(shots):
            #         self.qx.execute()
            #         # self.qx.execute(1)
            #         for i in range(n_qubits):
            #             c[i] = self.qx.get_measurement_outcome(i)
            #         idx = sum(v<<i for i, v in enumerate(c[::-1]))    
            #         p[idx] += 1/shots

            #     psgn = [1]
            #     for pt in wpp:
            #         if pt == "X":
            #             psgn = np.kron(psgn,xsgn)
            #         #elif pt == "Y":
            #         #    psgn = np.kron(psgn,xsgn) # TBD
            #         elif pt == "Z":
            #             psgn = np.kron(psgn,zsgn)
            #         else: # Identity
            #             psgn = np.kron(psgn,isgn)
            #     for pn in range(2**n_qubits):
            #         Epp += psgn[pn]*p[pn]                
            #     E += wsopp[wpp]*Epp
            #     self.expt += E

            #     if wpp == "I"*n_qubits:
            #         track_probs = p

            return E

        '''
        Callback function for accessing intermediate results of optimization
        '''
        def intermediate(cb):
            global track_opt
            print("Step: ",self.optstep)
            print("\tOptimal Parameters: ",cb)
            print("\tExpectation Value: ",self.expt)
            # print("\tOptimal Probabilities: ",self.probs)
            self.optstep += 1
            track_opt.append([self.optstep, cb, self.probs, self.expt])
            # input("Press Enter to continue to next step")
               
        if shots <= 0:
            args = [expectation_isv, params] 
        else:
            args = [expectation_qst, params]
           
        r = self.minimizer(*args, callback=intermediate, **self.minimizer_kwargs) # Run QAOA cycles
        return r

####################################################################################################
####################################################################################################
####################################################################################################    TSP
####################################################################################################
####################################################################################################

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

# Encoding      Semantics
# 001010100     2 > 1 > 0
# 001100010     1 > 2 > 0
# 010001100     2 > 0 > 1
# 010100001     1 > 0 > 2
# 100001010     0 > 2 > 1
# 100010001     0 > 1 > 2

def graph_problem():
    g = nx.Graph()
    g.add_edge(0,1,weight=1)
    g.add_edge(0,2,weight=1)
    g.add_edge(1,2,weight=1)
    return g

g = graph_problem()

####################################################################################################

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
    
    wsopp[Iall] = 1 # to get results

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

####################################################################################################

initstate = []
for i in range(0,len(g.nodes())**2): # Reference state preparation
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

    	elif i.count('Z') == 2: #check for the Iall case
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

####################################################################################################

maxiter = 4

qaoa_tsp_obj = QAOA(maxiter)
# res = qaoa_tsp_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)

####################################################################################################

# print(res.status, res.fun, res.x)
# # print(sum(track_opt[0][2])) # debug, should add up to almost 1

# find_solns = {}
# enc = 0
# for i in track_opt[-1][2]:
#     find_solns[format(enc,'#011b')] = i 
#     enc += 1

# plines = 10 # Top few solutions
# for elem in sorted(find_solns, key = find_solns.get, reverse = True) :
#     print(elem , " ::" , find_solns[elem])
#     plines -= 1
#     if plines == 0:
#         break

# # %matplotlib inline
# plt.ylim((0,1))
# plt.plot(track_opt[0][2],'--') # Initial
# plt.plot(track_opt[-1][2]) # Final
# plt.show()

####################################################################################################
####################################################################################################
####################################################################################################    MaxCut
####################################################################################################
####################################################################################################

# Wikipedia example

#     1--2
#    /|  |
#   0 |  |
#    \|  |
#     4--3 

# 43210
# 01010 - 10101
# 10100 - 01011
# 10, 11, 20, 21

# def graph_problem():
#     g = nx.Graph()
#     g.add_edge(0,1,weight=1)
#     g.add_edge(0,4,weight=1)
#     g.add_edge(1,2,weight=1)
#     g.add_edge(1,4,weight=1)
#     g.add_edge(2,3,weight=1)
#     g.add_edge(3,4,weight=1)
#     return g

# Square ring

#   0---1
#   |   |
#   3---2

def graph_problem():
    g = nx.Graph()
    g.add_edge(0,1)
    g.add_edge(0,3)
    g.add_edge(1,2)
    g.add_edge(2,3)
    return g

# Barbell Graph

#   0---1---2

# def graph_problem():
#     g = nx.Graph()
#     g.add_edge(0,1)
#     g.add_edge(1,2)
#     return g

g = graph_problem()

####################################################################################################

"""
Converts the MaxCut graph to wsopp for problem Hamiltonian for QUBO/Ising model
"""

def graph_to_wsopp_mxct(g):
    # for i in g.edges():
    #     print(i,g.edges[i]['weight'])
    wsopp = {}
    n_qubits = len(g.nodes())
    Iall = "I"*n_qubits

    for i,j in g.edges():
        # 0.5*Z_i*Z_j
        sopp = Iall[:n_qubits-1-i]+"Z"+Iall[n_qubits-1-i+1:]
        sopp = sopp[:n_qubits-1-j]+"Z"+sopp[n_qubits-1-j+1:]
        if sopp in wsopp:
            wsopp[sopp] = wsopp[sopp] + 0.5
        else:
            wsopp[sopp] = 0.5
        # -0.5*I_0
        if Iall in wsopp:
            wsopp[Iall] = wsopp[Iall] - 0.5
        else:
            wsopp[Iall] = -0.5
    return wsopp
    
wsopp = graph_to_wsopp_mxct(g)

# print(len(wsopp))
# for i in wsopp:
#     print(i,wsopp[i]) 

# {'IIIZZ': 0.5, 'IIIII': -3.0, 'ZIIIZ': 0.5, 'IIZZI': 0.5, 'ZIIZI': 0.5, 'ZZIII': 0.5, 'IZZII': 0.5}

####################################################################################################


initstate = []
for i in range(0,len(g.nodes())): # Reference state preparation
    initstate.append(("h",i))

# Refer: Rigetti --> Forest --> PyQuil --> paulis.py --> exponential_map()
def ansatz_pqasm_mxct(wsopp):
    return 0

def graph_pqasm_mxct(g):
    n_qubits = len(g.nodes())
    coeffs = [] # Weights for the angle parameter for each gate
    angles = [0,0] # Counts for [cost,mixing] Hamiltonian angles
    Iall = ""
    for i in range(n_qubits):
        Iall += "I"
    ansatz = [] # qasm tokens
    for i,j in g.edges():
        # 0.5*Z_i*Z_j
        ansatz.append(("cnot",[i,j]))
        ansatz.append(("rz",j))
        coeffs.append(2*0.5)
        angles[0] += 1 # gamma: cost Hamiltonian
        ansatz.append(("cnot",[i,j]))
        # -0.5*I_0
        ansatz.append(("x",0))
        ansatz.append(("rz",0))
        coeffs.append(-1*-0.5)
        angles[0] += 1 # gamma: cost Hamiltonian
        ansatz.append(("x",0))
        ansatz.append(("rz",0))
        coeffs.append(-1*-0.5)
        angles[0] += 1 # gamma: cost Hamiltonian
    for i in g.nodes():
        # -X_i
        ansatz.append(("h",i))
        ansatz.append(("rz",i))
        coeffs.append(2*-1)
        angles[1] += 1 # beta: mixing Hamiltonian
        ansatz.append(("h",i))
    return ansatz, coeffs, angles

ansatz, cfs, aid = graph_pqasm_mxct(g)

steps = 3 # Number of steps (QAOA blocks per iteration)

# Initial angle parameters for Hamiltonians cost (gammas) and mixing/driving (betas)
init_gammas = np.random.uniform(0, 2*np.pi, steps) 
init_betas = np.random.uniform(0, 2*np.pi, steps)


####################################################################################################

maxiter = 20

qaoa_mxct_obj = QAOA(maxiter)
res = qaoa_mxct_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)

####################################################################################################

print(res.status, res.fun, res.x)
# print(sum(track_opt[0][2])) # debug, should add up to almost 1

find_solns = {}
enc = 0

for i in track_opt[-1][2]:
    find_solns[format(enc,'#07b')] = i 
    enc += 1

plines = 6 # Top few solutions
for elem in sorted(find_solns, key = find_solns.get, reverse = True) :
    print(elem , " ::" , find_solns[elem])
    plines -= 1
    if plines == 0:
        break

# %matplotlib inline
plt.ylim((0,1))
plt.plot(track_opt[0][2],'--') # Initial
plt.plot(track_opt[-1][2]) # Final
plt.show()

####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
#
#                @@@@@@@   @@@  @@@   @@@@@@   @@@@@@@@  @@@  @@@  @@@  @@@  @@@
#                @@@@@@@@  @@@  @@@  @@@@@@@@  @@@@@@@@  @@@@ @@@  @@@  @@@  @@@
#                @@!  @@@  @@!  @@@  @@!  @@@  @@!       @@!@!@@@  @@!  @@!  !@@
#                !@!  @!@  !@!  @!@  !@!  @!@  !@!       !@!!@!@!  !@!  !@!  @!!
#                @!@@!@!   @!@!@!@!  @!@  !@!  @!!!:!    @!@ !!@!  !!@   !@@!@!
#                !!@!!!    !!!@!!!!  !@!  !!!  !!!!!:    !@!  !!!  !!!    @!!!
#                !!:       !!:  !!!  !!:  !!!  !!:       !!:  !!!  !!:   !: :!!
#                :!:       :!:  !:!  :!:  !:!  :!:       :!:  !:!  :!:  :!:  !:!
#                 ::       ::   :::  ::::: ::   :: ::::   ::   ::   ::   ::  :::
#                 :         :   : :   : :  :   : :: ::   ::    :   :     :   ::
#
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################
####################################################################################################