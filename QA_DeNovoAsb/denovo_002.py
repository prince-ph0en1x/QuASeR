"""
Read *.qubo file to form the Hamiltonian and solve the Ising and QUBO model using dimod exact solver

Hf = - 0.5*n0t0 + 1.0*n0t1 - 1.0*n0t0*n0t1

QUBO file content:

p qubo 0 2 2 1
n0t0 n0t0 -0.5
n0t1 n0t1 1.0
n0t0 n0t1 -1

Expected output:

{'n0t0': -1, 'n0t1': -1} -1.5
{'n0t0': 1, 'n0t1': -1} -0.5
{'n0t0': 1, 'n0t1': 1} -0.5
{'n0t0': -1, 'n0t1': 1} 2.5

{'n0t0': 1, 'n0t1': 0} -0.5
{'n0t0': 1, 'n0t1': 1} -0.5
{'n0t0': 0, 'n0t1': 0} 0.0
{'n0t0': 0, 'n0t1': 1} 1.0
"""

import dimod
solver = dimod.ExactSolver()

f = open("denovo_001.qubo", "r")
qubo_header = f.readline().split()
hii = {}
Jij = {}
Q = {}
for i in range(0,int(qubo_header[4])):
	x = f.readline().split()
	hii[x[0]] = float(x[2])	
	Q[(x[0],x[1])] = float(x[2])	
for i in range(0,int(qubo_header[5])):
	x = f.readline().split()
	Jij[(x[0],x[1])] = float(x[2])
	Q[(x[0],x[1])] = float(x[2])
f.close()

# print(hii, Jij)
response = solver.sample_ising(hii, Jij)
#Equivalent to: response = solver.sample_ising({'n0t0': -0.5, 'n0t1': 1.0}, {('n0t0', 'n0t1'): -1})

for sample, energy in response.data(['sample', 'energy']): print(sample, energy)

# print(Q)
response = solver.sample_qubo(Q)
#Equivalent to: response = solver.sample_qubo({'n0t0': -0.5, 'n0t1': 1.0}, {('n0t0', 'n0t1'): -1})

print()
for sample, energy in response.data(['sample', 'energy']): print(sample, energy)