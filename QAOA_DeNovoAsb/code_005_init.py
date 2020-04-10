from networkx import networkx as nx
import math
import numpy as np
import matplotlib.pyplot as plt
######################################################

"""
Defines the city graph for TSP: DNA example in paper
"""
def graph_problem():
    g = nx.DiGraph()
    g.add_edge(0,1,weight=7)
    g.add_edge(1,2,weight=7)
    g.add_edge(2,3,weight=7)
    g.add_edge(3,0,weight=9)
    g.add_edge(1,0,weight=3)
    g.add_edge(2,1,weight=3)
    g.add_edge(3,2,weight=3)
    g.add_edge(0,3,weight=1)
    g.add_edge(0,2,weight=4)
    g.add_edge(1,3,weight=4)
    g.add_edge(2,0,weight=6)
    g.add_edge(3,1,weight=6)
    return g
g = graph_problem()

######################################################

"""
Converts the TSP graph to wsopp for problem Hamiltonian for QUBO/Ising model
"""

n_cities = len(g.nodes())
n_timeslots = n_cities
n_qubits = n_cities*n_timeslots # MSQ = C0T0, LSQ = C3T3

def graph_to_wsopp_tsp(g):
    # for i in g.edges():
    #     print(i,g.edges[i]['weight'])
    wsopp = {}
    Iall = "I"*n_qubits
    
    penalty = 1e6 					# hyperparameter
    shift = 2*penalty*n_cities   	# hyperparameter
    
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

#####################################################

initstate = []
for i in range(0,n_qubits): # Reference state preparation
    initstate.append(("h",i))

"""
Converts the wsopp for problem Hamiltonian to the parametric ansatz
Logic reference: Rigetti --> Forest --> PyQuil --> paulis.py --> exponential_map()
"""
def ansatz_pqasm_tsp(wsopp):
    ansatz = [] # qasm tokens
    coeffs = [] # Weights for the angle parameter for each gate
    angles = [0,0] # Counts for [cost,mixing] Hamiltonian angles
    
    # Cost Hamiltonian

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

steps = 1 # Number of steps (QAOA blocks per iteration)

# Initial angle parameters for Hamiltonians cost (gammas) and mixing/driving (betas)
init_gammas = np.random.uniform(0, 2*np.pi, steps) 
init_betas = np.random.uniform(0, 2*np.pi, steps)

######################################################

import qxelarator
import re

qx = qxelarator.QX()
ptrn = re.compile('\(([+-]\d+.*\d*),([+-]\d+.*\d*)\)\s[|]([0-1]*)>') # Regular expression to extract the amplitudes from the string returned by get_state()

# self.minimizer = minimize
# self.minimizer_kwargs = {'method':'Nelder-Mead', 'callback':'intermediate', 'options':{'maxiter':maxiter, 
#                          'ftol':1.0e-6, 'xtol':1.0e-6, 'disp':True, 'return_all':True}}
p_name = "test_output/qaoa_run.qasm"
expt = 0  

def qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas):
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

    # print('>'*len(wsopp))
        

    def qasmify(params, wpp):
        prog = open(p_name,"w")
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

        prog.close()        

    def expectation_isv(params):
        # t = time.time()
        global ptrn
        E = 0
        expt = 0
        xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
        zsgn = [1,-1]
        isgn = [1,-1]
        global track_probs
        track_probs = np.zeros(2**n_qubits)

        for wpp in wsopp:
            print("#",end="",flush=True)
            qasmify(params,wpp)
            qx.set(p_name)

            Epp = 0
            p = np.zeros(2**n_qubits)
            qx.execute() 
            isv_str = qx.get_state()
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
            # expt += E

            if wpp == "I"*n_qubits:
                track_probs = p

            # print(expt)
            # break
            
        # print(E)  
        # self.elapsed = time.time() - t 
        return E

    def assignment(params):
        E = 0
        isgn = [1,-1]
        wpp = "I"*n_qubits
        qasmify(params,wpp)
        qx.set(p_name)

        Epp = 0
        p = np.zeros(2**n_qubits)
        qx.execute() 
        isv_str = qx.get_state()
        isv = re.findall(ptrn,isv_str)
        for basis in iter(isv):
            p[int(basis[2],2)] = float(basis[0])**2 + float(basis[1])**2 # Probability is square of modulus of complex amplitude
        psgn = [1]
        for pt in wpp:
            psgn = np.kron(psgn,isgn)
        for pn in range(2**n_qubits):
            Epp += psgn[pn]*p[pn]                
        E += 1*Epp
        # print(params,E,p)
        return [params,E,p]

    # f = open(logFile,"a")
    # f.write("Step 0: "+str(params)+"\n")
    # f.close()
    return assignment(params)
    # expectation_isv(params)
    # args = [expectation_isv, params]
    # r = self.minimizer(*args, **self.minimizer_kwargs) 
    # return r

######################################################

# Initial good guess
tries = 10
good_guess = []
avg_guess = []
bad_guess = []
good_E = 100000
avg_E = 100000
bad_E = -100000
plt_max = 0

for i in range(0,tries):
	init_gammas = np.random.uniform(0, 2*np.pi, steps) 
	init_betas = np.random.uniform(0, 2*np.pi, steps)
	res = qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
	print(res[1])
	if res[1] > bad_E:
		bad_E = res[1]
		bad_guess = res
	if abs(res[1]) < avg_E:
		avg_E = abs(res[1])
		avg_guess = res
	if res[1] < good_E:
		good_E = res[1]
		good_guess = res


if max(good_guess[2]) > plt_max:
	plt_max = max(good_guess[2])
if max(avg_guess[2]) > plt_max:
	plt_max = max(avg_guess[2])
if max(bad_guess[2]) > plt_max:
	plt_max = max(bad_guess[2])
# heights = [1, 2]

# f = plt.figure(constrained_layout=True)
# # f, (axt, axb) = plt.subplots(2, 1, sharex=True)

# import matplotlib.gridspec as gridspec
# spec = f.add_gridspec(ncols=1, nrows=2, height_ratios=heights)
# axt = f.add_subplot(spec[1, 1])
# axb = f.add_subplot(spec[1, 2])

plt.ylim((0,plt_max+0.001))
plt.plot(bad_guess[2],'bo') # Initial
plt.plot(good_guess[2],'gx') # Final
plt.plot(good_guess[2],'r+') # Final
plt.axvline(x = int('1000010000100001',2),color='m',linestyle='--')
plt.axvline(x = int('0100001000011000',2),color='m',linestyle='--')
plt.axvline(x = int('0010000110000100',2),color='m',linestyle='--')
plt.axvline(x = int('0001100001000010',2),color='m',linestyle='--')
plt.axvline(x = 65536 - int('1000010000100001',2),color='c',linestyle='--')
plt.axvline(x = 65536 - int('0100001000011000',2),color='c',linestyle='--')
plt.axvline(x = 65536 - int('0010000110000100',2),color='c',linestyle='--')
plt.axvline(x = 65536 - int('0001100001000010',2),color='c',linestyle='--')

# axt.set_ylim(.95, 1.)
# axb.set_ylim(0, plt_max+0.0001)  # most of the data
# axt.set_aspect(60000)
# axb.set_aspect(60000/plt_max)

# hide the spines between ax and ax2
# axt.spines['bottom'].set_visible(False)
# axb.spines['top'].set_visible(False)
# axt.xaxis.tick_top()
# axt.tick_params(labeltop=False)  # don't put tick labels at the top
# axb.xaxis.tick_bottom()

plt.show()

######################################################
# maxiter = 1

# from QAOA import QAOA, track_opt
# print(init_gammas, init_betas)
# qaoa_obj = QAOA(maxiter)
# for i in range(0,len(wsopp)):
#     print("#",end="")
# print()#, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# res = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# # print(res.status, res.fun, res.x)
# # res = qaoa_obj.qaoa_test(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
# # print(res)
# print(track_opt[-1])
# # # print(track_opt[-1])
# # # print(sum(track_opt[0][2]))
# # # # %matplotlib inline
# # # plt.ylim((0,1))
# # # plt.plot(track_opt[0][2],'--') # Initial
# # # plt.plot(track_opt[-1][2]) # Final
# # # plt.show()



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























