import numpy as np
import matplotlib.pyplot as plt
import ga
from ypstruct import structure  # Structure


"""Our cost finction"""


def myfunc(x):
    # return (sum(x**2)-(5*x))  # I added 2
    return sum(x**2)  # Original


# problem definition
problem = structure()
problem.costfunc = myfunc
problem.nvar = 5
problem.varmin = -100
problem.varmax = 100


# GA parameters
params = structure()
params.maxit = 100
params.npop = 50
params.pc = 1  # Whats this??
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
