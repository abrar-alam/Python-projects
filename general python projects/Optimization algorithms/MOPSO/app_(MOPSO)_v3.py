import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure  # Structure
from random import seed
from random import randint
from copy import deepcopy

# seed random number generator
seed(1)

"""Since this is a MOPSO algorithm, we define 2 (could be more) fitness functions"""
# We aim to maximize the first and minimize the second function


def sphere1(x):
    return (-1*sum(x**3)+(5*x))
    


def sphere2(x):
    return (1*sum(x**2)+(2*x))


# problem definition
# problem = structure()
costfunc1 = sphere1
costfunc2 = sphere2
nvar = 1
varmin = -10
varmax = 10
# Try to create another field so that I can store the solution vector, just as was done in matlab

# Iteration and population number parameters
maxit = 100
npop = 10

w = 1  # AKA Inertia coefficient
"""Damping Ratio of Inertia Coefficient. Its purpose is to reduce inertia coeff. while the algorithm iterates. If this
was not here, then the algo prematurely converges to a value that is not good enough"""

wdamp = 0.99
c1 = 2  # Personal acceleration coeff
c2 = 2  # Social or global acceleration coeff


# Initialization of swarms

# The particle template
empty_particle = structure()
empty_particle.Position = np.zeros(nvar)
empty_particle.Velocity = np.zeros(nvar)
# Since we have 2 fitness function, we must define 2 cost fields
empty_particle.Costs = (0,0)  # Its type is a tuple of floats
empty_particle.Current = structure()  #This structure stores current position and its associated costs
empty_particle.Current.Costs = (None, None)
empty_particle.Current.Position = np.zeros(nvar)
empty_particle.Bests = structure() # The first inner tuple represents the cost, second tuple represents the associated position
empty_particle.Bests.Costs = [(None, None)]
empty_particle.Bests.Positions = [np.zeros(nvar)]


# Now create the population array consisting of those partially initialized particles above
particle = [empty_particle.deepcopy() for _ in range(npop)]

# Now we initialize the global best cost as -infinity (becasue we aim to maximize our cost)
# Since this is MOPSO, GlobalBest acts as an external archive or a list
# Note that we are using list unlike PSO
GlobalBests = structure()
GlobalBests.Costs = [(np.array([-1*np.inf]), np.array([1*np.inf]))]
GlobalBests.Positions = [np.zeros(nvar)] 
# I had 'None' here before, we dont need to declare this
#GlobalBests.Positions = [np.zeros(nvar)]

# A helper function 
# The function below tests whether the first arg dominates the second arg
def dominates(T1: tuple, T2: tuple) -> bool:
    if (((T1[0][0] >= T2[0][0]) and (T1[1][0] <= T2[1][0])) and ((T1[0][0] > T2[0][0]) or (T1[1][0] < T2[1][0]))):
        return True
    return False


# a function that performs the non-dominated sorting, and updated the GlobalBests
# This method returns 2 lists: 1. Updated global bests costs 2. Updated global bests positions

def update_global_bests(g_bests_costs:list, g_bests_positions:list, p_bests_costs: list, p_bests_positions:list):
    g_bests_costs1 = g_bests_costs
    g_bests_positions1 = g_bests_positions
    p_bests_costs1 = p_bests_costs
    p_bests_positions1 = p_bests_positions
    
    for i in range(len(p_bests_costs1)):
        append_flag = False
        """A flag below that indicates all the costs tuple are incomparable 
            with the personal best tuple in this iteration"""
        all_non_dominated = False 
        index = -1
        for s in g_bests_costs: # Here I made the correction (I am referring 'g_bests' instead of 'g_best1')
            index = index + 1
            if (dominates(p_bests_costs1[i],  s) \
                or (np.array_equal(s[0], p_bests_costs1[i][0]) and np.array_equal(s[1], p_bests_costs1[i][1]))):# If p_bests[i].Cost dominates s.Cost
                #index = index + 1
                g_bests_costs1.remove(s)
                del g_bests_positions1[index] # This is to delete the associated position
                index = index-1
                #g_bests_positions1.remove(t)
                # Here add some codes to take removal of associated positions into account
                print("Print immidiately after removal") # For debugging
                append_flag=True
                all_non_dominated = False
                
            elif dominates(s, p_bests_costs1[i]):
                break # I added 'break' instead of 'pass'
            
            else:
                all_non_dominated = True
                
        if (append_flag or all_non_dominated):
            g_bests_costs1 = g_bests_costs1+[p_bests_costs1[i]]
            g_bests_positions1 = g_bests_positions1 + [p_bests_positions1[i]]
        
        g_bests_costs = deepcopy(g_bests_costs1)
        g_bests_positions = deepcopy(g_bests_positions1)
    return g_bests_costs1, g_bests_positions1  

# Another totally similar function that updates the each particle's personal bests
def update_personal_bests(p_bests_costs:list, p_bests_positions:list, current_costs: list, current_position:list):
    p_bests_costs1 = p_bests_costs
    p_bests_positions1 = p_bests_positions
    current_costs1 = current_costs
    current_position1 = current_position
    
    for i in range(len(current_costs1)):
        append_flag = False
        """A flag below that indicates all the costs tuple are incomparable 
            with the personal best tuple in this iteration"""
        all_non_dominated = False 
        index = -1
        for s in p_bests_costs: # Here I made the correction (I am referring 'g_bests' instead of 'g_best1')
            index = index + 1
            if (dominates(current_costs1[i],  s) \
                or (np.array_equal(s[0],  current_costs1[i][0]) and np.array_equal(s[1], current_costs1[i][1]))):# If p_bests[i].Cost dominates s.Cost
                #index = index + 1
                p_bests_costs1.remove(s)
                del p_bests_positions1[index] # This is to delete the associated position
                index = index-1
                #g_bests_positions1.remove(t)
                # Here add some codes to take removal of associated positions into account
                print("Print immidiately after removal") # For debugging
                append_flag=True
                all_non_dominated = False
                
            elif dominates(s, current_costs1[i]):
                break # I added 'break' instead of 'pass'
            else:
                all_non_dominated = True
                
        if (append_flag or all_non_dominated):
            p_bests_costs1 = p_bests_costs1+[current_costs1[i]]
            p_bests_positions1 = p_bests_positions1 + [current_position1[i]]
        
        p_bests_costs = deepcopy(p_bests_costs1)
        p_bests_positions = deepcopy(p_bests_positions1)
    return p_bests_costs1, p_bests_positions1 

# Initialize population members
for i in range(npop):
    # Generate random solution
    particle[i].Position = np.random.uniform(
        low=varmin, high=varmax, size=nvar)

    # Initialize velocity
    particle[i].Velocity = np.zeros(nvar)

    # Evaluations
    particle[i].Costs = (costfunc1(particle[i].Position), costfunc2(particle[i].Position))
    
    particle[i].Bests.Costs = [deepcopy(particle[i].Costs)] # We cast these as a list to prevent index error
    particle[i].Bests.Positions = [deepcopy(particle[i].Position)]

    # Update global bests by performing non-dominated sorting
    """if particle[i].Best.Cost > GlobalBest.Cost:
        GlobalBest = particle[i].Best"""
    # update is method to be defined, which performs non-dominated sorting
    # We are basically initializing leaders in the external archive here
    GlobalBests.Costs, GlobalBests.Positions = update_global_bests(GlobalBests.Costs,GlobalBests.Positions, particle[i].Bests.Costs,particle[i].Bests.Positions)

# We are not implementing Quality(leaders) functionality because we aim, to randomly select a leader for each particle


# Array to hold the best cost of each iteration
#BestCosts = np.zeros(maxit) I commented out recently

# main loop of PSO
for it in range(0, maxit):
    for i in range(0, npop):
        """We are adding the leader selection here (based on randomization), which is a characteristic of MOPSO"""
        if (not (len(GlobalBests.Costs) == 0)):

            value = randint(0, len(GlobalBests.Costs)-1) # Index of the global leader for this particle

            
            # Update velocity
            particle[i].Velocity = w*particle[i].Velocity + c1 * np.random.uniform(0, 1) * (particle[i].Bests.Positions[randint(0, len(particle[i].Bests.Positions)-1)] - particle[i].Position)\
            + c2 * np.random.uniform(0, 1)\
            * (GlobalBests.Positions[value] - particle[i].Position)

            # Update position
            particle[i].Position = particle[i].Position + particle[i].Velocity

            # Now mutation (due to it being MOPSO). We will implement this later
        else:
            # I will add my code here later 
            pass

        # Modification to compensate for the boundary bug
        # Clipping the max value
        particle[i].Position = np.minimum(
            particle[i].Position, np.array([varmax for _ in range(0, nvar)]))
        # Now clipping for the min value
        particle[i].Position = np.maximum(
            particle[i].Position, np.array([varmin for _ in range(0, nvar)]))

        # Evaluation
        particle[i].Cost = (costfunc1(particle[i].Position), costfunc2(particle[i].Position))
        # Now update the 'Current' attrib which stores current position and costs
        particle[i].Current.Costs = deepcopy(particle[i].Cost)
        particle[i].Current.Position = deepcopy(particle[i].Position)
        # Update personal best
        """if particle[i].Cost > particle[i].Best.Cost:
            particle[i].Best.Position = particle[i].Position
            particle[i].Best.Cost = particle[i].Cost

            if particle[i].Best.Cost > GlobalBest.Cost:
                GlobalBest = particle[i].Best"""
        # Now update personal and global bests
        particle[i].Bests.Costs,particle[i].Bests.Positions  = update_personal_bests(particle[i].Bests.Costs,particle[i].Bests.Positions,[particle[i].Current.Costs], [particle[i].Current.Position] ) # A function we will define
        GlobalBests.Costs , GlobalBests.Positions = update_global_bests(GlobalBests.Costs,GlobalBests.Positions,particle[i].Bests.Costs,particle[i].Bests.Positions )
    # Store the best cost value
    #BestCosts[it] = GlobalBest.Cost

    # Print iteration info.
    #print("Iteration no. " + str(it+1) + ", Best cost = "+str(BestCosts[it]))

    # Dampling the inertia Coefficient. This line actually reduces the inertia coeff. as this algo iterates
    w *= wdamp
    print("Iteration "+str(it) + " is completed.")


# Plotting the progress
"""plt.semilogy(BestCosts)
plt.xlim(0, maxit)
plt.xlabel('Iteration')
plt.ylabel('Best cost')
plt.title('PSO')
plt.grid(True)
plt.show()"""
#print(GlobalBests.Costs)
print(len(GlobalBests.Positions))
print(len(GlobalBests.Costs))