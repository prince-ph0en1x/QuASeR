# Check larger instances of Max-Cut with more steps and compressed wsopp
# ..:$ python3 code_003.py 

import networkx as nx
import numpy as np
from scipy.optimize import minimize
from qxelarator import qxelarator
import matplotlib.pyplot as plt
import re

######################################################

ptrn = re.compile('\(([+-]\d+.*\d*),([+-]\d+.*\d*)\)\s[|]([0-1]*)>')
isv_prob = True

track_opt = []
track_optstep = 0
track_probs = []

class QAOA(object):

    def __init__(self, maxiter, shots):
        self.qx = qxelarator.QX()
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'options':{'maxiter':maxiter, 
                                 'ftol':1.0e-6, 'xtol':1.0e-6, 'disp':True, 'return_all':True}}
        self.p_name = "test_output/qaoa_run.qasm"
        self.shots = shots 
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
                for i in range(self.shots):
                    self.qx.execute()
                    # self.qx.execute(1)
                    for i in range(n_qubits):
                        c[i] = self.qx.get_measurement_outcome(i)
                    idx = sum(v<<i for i, v in enumerate(c[::-1]))    
                    p[idx] += 1/self.shots

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
            print("Current Optimal Parameters: ",cb)
            print("Current Expectation Value: ",self.expt)
            print("Current Optimal Probabilities: ",track_probs)
            track_optstep += 1
            input("Press Enter to continue to step "+str(track_optstep))
            track_opt.append([track_optstep, cb, track_probs])
               
        args = [expectation_isv, params]
        r = self.minimizer(*args, callback=intermediate, **self.minimizer_kwargs) 
        return r

######################################################

#     1--2
#    /|  |
#   0 |  |
#    \|  |
#     4--3 

# 43210
# 01010 - 10101
# 10100 - 01011
# 10, 11, 20, 21

def graph_problem():
    g = nx.Graph()
    g.add_edge(0,1)
    g.add_edge(0,4)
    g.add_edge(1,2)
    g.add_edge(1,4)
    g.add_edge(2,3)
    g.add_edge(3,4)
    return g

g = graph_problem()

print(g)

######################################################

def graph_to_wsopp(g, n_qubits):
    wsopp = {}
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
    
wsopp = graph_to_wsopp(g, len(g.nodes()))
# {'IIIZZ': 0.5, 'IIIII': -3.0, 'ZIIIZ': 0.5, 'IIZZI': 0.5, 'ZIIZI': 0.5, 'ZZIII': 0.5, 'IZZII': 0.5}

######################################################

initstate = []
for i in range(0,len(g.nodes())): # Reference state preparation
    initstate.append(("h",i))

def graph_to_pqasm(g,n_qubits):
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

ansatz, cfs, aid = graph_to_pqasm(g,len(g.nodes()))

steps = 2 # number of steps (QAOA blocks per iteration)

# Initial angle parameters for Hamiltonians cost (gammas) and mixing/driving (betas)

init_gammas = np.random.uniform(0, 2*np.pi, steps) 
init_betas = np.random.uniform(0, np.pi, steps)

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

maxiter = 5#20
shots = 100 # should be some factor of number of qubits to have the same precision

qaoa_obj = QAOA(maxiter, shots)
res = qaoa_obj.qaoa_run(wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas)
print(res.status, res.fun, res.x)
print(track_opt[-1])
print(sum(track_opt[0][2]))
# %matplotlib inline
plt.ylim((0,1))
plt.plot(track_opt[0][2],'--') # Initial
plt.plot(track_opt[-1][2]) # Final
plt.show()