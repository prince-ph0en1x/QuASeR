import numpy as np
from scipy.optimize import minimize
import qxelarator
import re
import time

######################################################

ptrn = re.compile('\(([+-]\d+.*\d*),([+-]\d+.*\d*)\)\s[|]([0-1]*)>') # Regular expression to extract the amplitudes from the string returned by get_state()
isv_prob = True # True if the internal state vector is accessed using get_state() instead of measurement aggregates over multiple shots
shots = 500 # If isv_prob is false, the experiment is run over multple shots for measurement aggregate. Should be some factor of number of qubits to have the same precision

track_opt = []
track_optstep = 0
track_probs = []

logFile = "runLog.txt"

class QAOA(object):

    def __init__(self, maxiter):
        self.qx = qxelarator.QX()
        self.minimizer = minimize
        self.minimizer_kwargs = {'method':'Nelder-Mead', 'callback':'intermediate', 'options':{'maxiter':maxiter, 
                                 'ftol':1.0e-6, 'xtol':1.0e-6, 'disp':True, 'return_all':True}}
        self.p_name = "test_output/qaoa_run.qasm"
        self.expt = 0  
        self.elapsed = 0
    
    def intermediate(cb, state):
        global track_opt
        global track_optstep
        global track_probs
        print("hi")

        f = open(logFile,"a")
        f.write("Step "+str(track_optstep)+": "+"\n")
        f.write("Current Optimal Parameters: "+str(cb)+"\n")
        f.write("Current Expectation Value: "+str(self.expt)+"\n")
        f.write("Current Optimal Probabilities: "+str(track_probs)+"\n")
        f.write("Time (isv): "+str(self.elapsed)+"\n")
        f.flush()
        f.close()

        # print("Step: ",track_optstep)
        # print("Current Optimal Parameters: ",cb)
        # print("Current Expectation Value: ",self.expt)
        # print("Current Optimal Probabilities: ",track_probs)
        track_optstep += 1
        input("Press Enter to continue to step "+str(track_optstep))
        track_opt.append([track_optstep, cb, track_probs])

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
            t = time.time()
            global ptrn
            E = 0
            self.expt = 0
            xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
            zsgn = [1,-1]
            isgn = [1,-1]
            global track_probs
            track_probs = np.zeros(2**n_qubits)

            for wpp in wsopp:
                print("#",end="",flush=True)
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
            print()  
            self.elapsed = time.time() - t 
            return E

        f = open(logFile,"a")
        f.write("Step 0: "+str(params)+"\n")
        f.close()

        args = [expectation_isv, params]
        r = self.minimizer(*args, **self.minimizer_kwargs) 
        return r












    # def qaoa_test(self, wsopp, initstate, ansatz, cfs, aid, steps, init_gammas, init_betas):
    #     print("QAOA Test")
    #     n_qubits = len(list(wsopp.keys())[0])
    #     pqasm = []
    #     coeffs = []
    #     ang_nos = []
    #     params = []
    #     for gate in initstate:
    #         pqasm.append(gate)
    #     for p in range(0,steps):
    #         for gate in ansatz:
    #             pqasm.append(gate)
    #         coeffs = np.hstack((coeffs,cfs))
    #         ang_nos = np.hstack((ang_nos,aid))
    #         params.append(init_gammas[p])
    #         params.append(init_betas[p]) 

    #     def qasmify(params, wpp):
    #         global isv_prob
    #         prog = open(self.p_name,"w")
    #         prog.write("# De-parameterized QAOA ansatz\n")
    #         prog.write("version 1.0\n")
    #         prog.write("qubits "+str(n_qubits)+"\n")
            
    #         prog.write(".qk(1)\n")
            
    #         # De-parameterize pqasm
    #         a_id = 0
    #         a_ctr = 0
    #         c_ctr = 0
    #         for i in pqasm:
    #             # 1-qubit parametric gates
    #             if i[0] == 'rx' or i[0] == 'ry' or i[0] == 'rz':
    #                 prog.write(i[0]+" q["+str(i[1])+"],"+str(coeffs[c_ctr]*params[a_id])+"\n")
    #                 c_ctr += 1
    #                 a_ctr += 1
    #                 if a_ctr >= ang_nos[a_id]:
    #                     a_id += 1
    #                     a_ctr = 0
    #             # 1-qubit discrete gates
    #             elif i[0] == 'x' or i[0] == 'y' or i[0] == 'z' or i[0] == 'h':
    #                 prog.write(i[0]+" q["+str(i[1])+"]\n")
    #             # 2-qubit discrete gates
    #             else:
    #                 prog.write(i[0]+" q["+str(i[1][0])+"],q["+str(i[1][1])+"]\n")
            
    #         # Pre-rotation for Z-basis measurement
    #         tgt = n_qubits-1
    #         for pt in wpp:
    #             if pt == "X":
    #                 prog.write("ry q"+str(tgt)+",1.5708\n")
    #             elif pt == "Y":
    #                 prog.write("rx q"+str(tgt)+",-1.5708\n")
    #             # else Z or Identity
    #             tgt -= 1

    #         # Measure all
    #         if not isv_prob:
    #             for i in range(n_qubits):
    #                 prog.write("measure q["+str(i)+"]\n")

    #         prog.close()        

    #     def expectation_isv(params):
    #         global ptrn
    #         E = 0
    #         self.expt = 0
    #         xsgn = [-1,1] # Try [1,-1] with ry +pi/2 in qasmify for pt == 'X'
    #         zsgn = [1,-1]
    #         isgn = [1,-1]
    #         global track_probs
    #         track_probs = np.zeros(2**n_qubits)

    #         for wpp in wsopp:
    #             print("#",end="",flush=True)
    #             qasmify(params,wpp)
    #             self.qx.set(self.p_name)

    #             Epp = 0
    #             p = np.zeros(2**n_qubits)
    #             self.qx.execute() 
    #             isv_str = self.qx.get_state()
    #             isv = re.findall(ptrn,isv_str)
    #             for basis in iter(isv):
    #                 p[int(basis[2],2)] = float(basis[0])**2 + float(basis[1])**2 # Probability is square of modulus of complex amplitude
                
    #             psgn = [1]
    #             for pt in wpp:
    #                 if pt == "X":
    #                     psgn = np.kron(psgn,xsgn)
    #                 #elif pt == "Y":
    #                 #    psgn = np.kron(psgn,xsgn) # TBD
    #                 elif pt == "Z":
    #                     psgn = np.kron(psgn,zsgn)
    #                 else: # Identity
    #                     psgn = np.kron(psgn,isgn)
    #             for pn in range(2**n_qubits):
    #                 Epp += psgn[pn]*p[pn]                
    #             E += wsopp[wpp]*Epp
    #             self.expt += E

    #             if wpp == "I"*n_qubits:
    #                 track_probs = p
    #         print()   
    #         return E

    #     return expectation_isv(params)
