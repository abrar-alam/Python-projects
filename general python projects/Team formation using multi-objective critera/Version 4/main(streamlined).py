import random
from teams import Teams
from team import Team
from user import User
import copy as cp

# These are the important global variables
"""Important Note:  'total_num_of_people' variable below MUST be divisble by 'number_of_teams' variable. 
   To put simply, we only focus on generating multiple SAME sized teams"""
number_of_teams = 10  # These values can be changed as long as we adhere to the note above
# These values can be changed as long as we adhere to the note above
total_num_of_people = 50
gender_diversity = True
availability_diversity = True
personality_diversity = False
skills_diversity = False
u = [User() for t in range(0, total_num_of_people)
     ]  # This creates random users and puts them in a list. An important variable
# users now have been created
# Now we execute the combinatorics toolset


# I changed this function to return a set


def possible_teams(num_of_people: int, num_of_teams: int) -> list:
    """Algorithm:
        1. Create or define an empty list called "possible results"
        1. Create a list with the length = num of people, and each element is the user id
        2. make a copy of the list above, name it as temp2
        3. for numberofpeople C (numberof people/number of teams) times:
            a. Shuffle the temp array
            b. Declare a temp list to hold a configuration to be computed, named alpha
            c. for number of team times:

                i. for number of people within a team times:
                    I. Pick a random element from temp.
                    II. Add that randomly picked element in a temp list, called tau
                    III. Delete that random element from the temp list 

                ii. Convert list tau into a tuple, then append that tuple to alpha 
            d. Then append alpha into "possible results" list 
            e. reset or clear the variable "alpha" """

    possible_result = []
    user_id_list = [t for t in range(1, num_of_people+1)]
    temp2 = cp.deepcopy(user_id_list)
    for i in range(0, 200):
        random.shuffle(temp2)
        temp3 = cp.deepcopy(temp2)
        alpha = []
        for j in range(0, num_of_teams):
            tau = []
            for k in range(0, num_of_people//num_of_teams):
                t1 = random.choice(temp3)
                tau.append(t1)
                temp3.remove(t1)
            alpha.append(convert(tau))
        possible_result.append(alpha)
    return possible_result

# Python3 program to convert a
# list into a tuple


def convert(list):
    return (*list, )


all_possible_combination = possible_teams(total_num_of_people, number_of_teams)
# print(possible_teams(6, 2))  # This shows that the above method works!!
# Now we gotta create
"""my_list = [[('1', '2', '3'), ('4', '5', '6')], [
    ('4', '5', '6'), ('1', '2', '3')]]
l1 = [('1', '2', '3'), ('4', '5', '6')]
l2 = [('4', '5', '6'), ('1', '2', '3')]

l1.sort()
l2.sort()
print(l1)
print(l2)
my_set = set()
my_set.add(tuple(l1))
print(my_set)
my_set.add(tuple(l2))
print(my_set)
l3 = [('4', '5', '6'), ('1', '2', '3')]
l3 = tuple(l3)
print(l3)
if(l3 in my_set):
    print("True")"""

# Now lets create a list of possible teams (object)
possible_teams = []
# The variable above shows the format as follows:
# [[u1,u2,u3],[u4,u5,u6],[u1,u5,u6],[u2,u3,u4],..........]
"""for i in range(0, len(all_possible_combination)):
    for j in range(0, number_of_teams):
        people_list = []
        for k in range(0, total_num_of_people//number_of_teams):
            temp_string = all_possible_combination[i][j][k]
            exec("people_list.append(u%s)" % (temp_string))
            #exec("%s = User(%d, g6, p6)" % (name, 1))
        possible_teams.append(people_list)"""

for i in range(0, len(all_possible_combination)):
    for j in range(0, number_of_teams):
        people_list = []
        for k in range(0, total_num_of_people//number_of_teams):
            temp_string = all_possible_combination[i][j][k]
            exec("people_list.append(u[%d-1])" % (temp_string))
            #exec("%s = User(%d, g6, p6)" % (name, 1))
        possible_teams.append(people_list)
# print(possible_teams)  # for debugging
"""for team in possible_teams:
    for user in team:
        print(user.toString())
    print("\n")"""
# resume from here
# the variable below stores all the possible teams (without distinguishing among configs)
list_of_team = []  # An important variable!!


for i in range(0, len(possible_teams), number_of_teams):
    #temp = Team(possible_teams[i])
    #temp1 = Team(possible_teams[i+1])
    # list_of_team.append(temp)
    # list_of_team.append(temp1)
    # My mod code starts from here (to make things dynamic)
    t = i
    for j in range(0, number_of_teams):
        list_of_team.append(Team(
            possible_teams[t], gender_diversity, personality_diversity, skills_diversity, availability_diversity))
        t = t+1


# Check to see that we indeed have added all the teams
"""for team in list_of_team:  # This works fine!!
    print(team.toString())"""

# Now we for each configuration from the overall list of team AKA 'list_of_team'
# An important variable, where each elemnt is a single unique configuration, or a list of 'Teams' object
list_of_configurations = []  # An important variable

for i in range(0, len(list_of_team), number_of_teams):
    temp = []
    # temp.append(list_of_team[i])
    # temp.append(list_of_team[i+1])
    #temp1 = Teams(i//2, temp)
    # list_of_configurations.append(temp1)
    # My added modifications to make these dynamic
    for j in range(i, i+number_of_teams):
        temp.append(list_of_team[j])
    temp1 = Teams(i//number_of_teams, temp)
    list_of_configurations.append(temp1)

"""for config in list_of_configurations:  # this shows that nothing is wrong so far
    print(config.toString())"""
# print(list_of_configurations[0].to)

# Now we store performance and fairness scores associated with each configuration
raw_objective_values = []  # An important variable!

# the method below builds 'raw_objective_values'


def build_objective_val(x: list):
    for config in list_of_configurations:
        x.append((config.performance_score(), config.fairness_score()))


# Now we build 'raw objective values'
build_objective_val(raw_objective_values)  # Very important!!

# print(raw_objective_values)  # Works fine!

# print(len(raw_objective_values))  # works fine!

# Now we work on findinf non-dominated set
# an important variable that stores which indices  of the 'raw_objective_values'correspond to non dominated front
non_dominated_sols = []  # an important variable!

# The method below will modify the 'non_dominated_sols'


def find_non_dominated_set(ls: list) -> list:
    # Assume we have all 20 configs as non-dominated members (for the case when we have 6 people, and we want 2 teams)
    tmp = [x for x in range(0, len(all_possible_combination))]

    for i in range(0, len(ls)):
        for j in range(0, len(ls)):
            if(dominates(ls[j], ls[i])):  # If the first arg dominates second arg
                if (i in tmp):
                    tmp.remove(i)
                break

    return tmp


# The function below tests whether the first arg dominates the second arg
# Note: We want to maximize performance and minimize fairness values
def dominates(T1: tuple, T2: tuple) -> bool:
    if (((T1[0] >= T2[0]) and (T1[1] <= T2[1])) and ((T1[0] > T2[0]) or (T1[1] < T2[1]))):
        return True
    return False


non_dominated_sols = find_non_dominated_set(
    raw_objective_values)  # Important line

# print(non_dominated_sols)  # Works fine!!

# Now we print out the non-dominated raw objective values to varify correctness
# This shows that we have correct results but have some duplicates!
"""for num in non_dominated_sols:
    print(raw_objective_values[num])"""

# Now we form a new refined list of configuration with only non-duplicate pareto optimal solutions
final_configs = []  # An importnat variable. a list of 'Teams' objects

# the method below takes 'non_dominated_sols' as argument, and it modifies 'final_configs'


def generate_final_solution(l: list) -> list:
    temp = [raw_objective_values[l[0]]]
    temp2 = [l[0]]
    for num in l:
        if (raw_objective_values[num] not in temp):
            temp.append(raw_objective_values[num])
            temp2.append(num)

    final_configs = [list_of_configurations[x] for x in temp2]
    return final_configs


# The variable below is a list of 'Teams' object
final_configs = generate_final_solution(non_dominated_sols)

# My added to varify
for teams in final_configs:
    print(teams.toString())

for config in final_configs:
    print(str(config.performance_score())+"\t" + str(config.fairness_score()))

for user in u:
    print(user)
