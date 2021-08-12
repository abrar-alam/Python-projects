import numpy as np
import matplotlib.pyplot as plt
#import ga
from ypstruct import structure  # Structure
import copy as cp


"""Our cost finction"""


def sphere(x):
    return (-1*sum(x**2)+(5*x))  # I added 2
    # return sum(x**2)  # Original


# problem definition
#problem = structure()
costfunc = sphere
nvar = 1
varmin = -100  # A possible bug here, changing it introduces a bug
varmax = 100
# Try to create another field so that I can store the solution vector, just as was done in matlab


# GA parameters
#params = structure()
maxit = 1000
npop = 50

w = 1  # AKA Inertia coefficient
"""Damping Ratio of Inertia Coefficient. Its purpose is to reduce inertia coeff. while the algorithm iterates. If this 
was not here, then the algo prematurely converges to a value that is not good enough"""

wdamp = 0.99
c1 = 2  # Personal acceleration coeff
c2 = 2  # Social or global acceleration coeff
"""# Run GA
out = ga.run(problem, params)

# Results
# plt.plot(out.bestcost)
plt.semilogy(out.bestcost)
plt.xlim(0, params.maxit)
plt.xlabel('Iteration')
plt.ylabel('Best cost')
plt.title('GA')
plt.grid(True)
plt.show()"""


# Initialization of swarms

# The particle template
empty_particle = structure()
empty_particle.Position = np.zeros(nvar)
empty_particle.Velocity = np.zeros(nvar)
empty_particle.Cost = None  # Its type is float
empty_particle.Best = structure()
empty_particle.Best.Position = np.zeros(nvar)
empty_particle.Best.Cost = None  # Its type is also float

# Now create the population array consisting of those partially initialized particles above
particle = [empty_particle.deepcopy() for _ in range(npop)]

# Now we initialize the global best cost as -infinity (becasue we aim to maximize our cost)
GlobalBest = structure()
GlobalBest.Cost = -1*np.inf
# I had 'None' here before, we dont need to declare this
GlobalBest.Position = np.zeros(nvar)

# Initialize population members
for i in range(npop):
    # Generate random solution
    particle[i].Position = np.random.uniform(
        low=varmin, high=varmax, size=nvar)

    # Initialize velocity
    particle[i].Velocity = np.zeros(nvar)

    # Evaluation
    particle[i].Cost = costfunc(particle[i].Position)

    # Update the personal best
    particle[i].Best.Position = particle[i].Position
    particle[i].Best.Cost = particle[i].Cost

    # Update global best
    if particle[i].Best.Cost > GlobalBest.Cost:
        GlobalBest = particle[i].Best

# Array to hold the best cost of each iteration
BestCosts = np.zeros(maxit)

# main loop of PSO
for it in range(0, maxit):
    for i in range(0, npop):
        # Update velocity
        particle[i].Velocity = w*particle[i].Velocity + c1 * np.random.uniform(0, 1) * (particle[i].Best.Position - particle[i].Position)\
            + c2 * np.random.uniform(0, 1)\
            * (GlobalBest.Position - particle[i].Position)

        # Update position
        particle[i].Position = particle[i].Position + particle[i].Velocity

        # Modification to compensate for the boundary bug
        # Clipping the max value
        particle[i].Position = np.minimum(
            particle[i].Position, np.array([varmax for _ in range(0, nvar)]))
        # Now clipping for the min value
        particle[i].Position = np.maximum(
            particle[i].Position, np.array([varmin for _ in range(0, nvar)]))

        # Evaluation
        particle[i].Cost = costfunc(particle[i].Position)

        # Update personal best
        if particle[i].Cost > particle[i].Best.Cost:
            particle[i].Best.Position = particle[i].Position
            particle[i].Best.Cost = particle[i].Cost

            if particle[i].Best.Cost > GlobalBest.Cost:
                GlobalBest = particle[i].Best

    # Store the best cost value
    BestCosts[it] = GlobalBest.Cost

    # Print iteration info.
    print("Iteration no. " + str(it+1) + ", Best cost = "+str(BestCosts[it]))

    # Dampling the inertia Coefficient. This line actually reduces the inertia coeff. as this algo iterates
    w *= wdamp


# Plotting the progress
plt.semilogy(BestCosts)
plt.xlim(0, maxit)
plt.xlabel('Iteration')
plt.ylabel('Best cost')
plt.title('PSO')
plt.grid(True)
plt.show()
