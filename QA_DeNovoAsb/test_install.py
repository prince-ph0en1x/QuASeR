import dimod
solver = dimod.ExactSolver()
response = solver.sample_ising({'a': -0.5, 'b': 1.0}, {('a', 'b'): -1})
for sample, energy in response.data(['sample', 'energy']): print(sample, energy)

'''
Hf = - 0.5a + 1.0b - 1.0ab

{'a': -1, 'b': -1} -1.5
{'a': 1, 'b': -1} -0.5
{'a': 1, 'b': 1} -0.5
{'a': -1, 'b': 1} 2.5
'''