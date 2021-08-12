import numpy as np
import matplotlib.pyplot as plt
import ga
from ypstruct import structure  # Structure


"""Our cost finction"""


def sphere(x):
    # return (sum(x**2)-(5*x))  # I added 2
    return sum(x**2)  # Original


# problem definition
problem = structure()
problem.costfunc = sphere
problem.nvar = 5  # Number of variable
# Important: We can also assign a list with different lower bounds for different variables (Multivariable case only)
problem.varmin = -10
# Important: We can also assign a list with different upper bounds for different variables (Multivariable case only)
problem.varmax = 10


# GA parameters
params = structure()
params.maxit = 100  # Max. number of iteratuons to be performed
params.npop = 50  # initial population size
params.beta = 1  # Selection pressure
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
