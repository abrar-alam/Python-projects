import numpy as np
import matplotlib.pyplot as plt
import ga
from ypstruct import structure  # Structure


"""Our cost finction"""


def myfunc(x):
    return (-1*sum(x**2)+(5*x))  # I added 2
    # return sum(x**2)  # Original


# problem definition
problem = structure()
problem.costfunc = myfunc
problem.nvar = 1
problem.varmin = 4
problem.varmax = 6


# GA parameters
params = structure()
params.maxit = 100
params.npop = 50
"""pc is the proportion quantifier: eg. say
   pc = 1 means, my population size, npop and number of children, nc are eqal in size
   pc =2 means number of children is twice as much as the ppopulation size, npop"""
params.pc = 1
params.gamma = 0.1  # Enables the algorithm to explore the serachspace efficiently
params.mu = 0.01  # Mutation rate
params.sigma = 0.1


# Run GA
out = ga.run(problem, params)

# Results
# plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iteration')
plt.ylabel('Best cost')
plt.title('GA')
plt.grid(True)
plt.show()
