from teams import Teams
from personality import Personality
from gender import Gender
from user import User
from team import Team
import statistics as st
import itertools
import numpy as np

# These are the important global variables
number_of_teams = 4
total_num_of_people = 8

# Say we have 6 persons, we want to generate 2 teams

# first we create 6 users
p1 = Personality(20, 22, 33, 44, 66)
g1 = Gender(1)
u1 = User(1, g1, p1)

p2 = Personality(10, 10, 50, 30, 70)
g2 = Gender(-1)
u2 = User(2, g2, p2)

p3 = Personality(50, 20, 30, 40, 80)
g3 = Gender(1)
u3 = User(3, g3, p3)

p4 = Personality(42, 55, 23, 44, 76)
g4 = Gender(-1)
u4 = User(4, g4, p4)

p5 = Personality(12, 55, 33, 44, 66)
g5 = Gender(-1)
u5 = User(5, g5, p5)

p6 = Personality(32, 85, 33, 74, 66)
g6 = Gender(-1)
u6 = User(6, g6, p6)

p7 = Personality(22, 34, 90, 54, 60)
g7 = Gender(1)
u7 = User(7, g7, p7)

p8 = Personality(2, 50, 85, 40, 50)
g8 = Gender(-1)
u8 = User(8, g8, p8)

# 6 users now have been created
# Now we execute the combinatorics toolset


# I changed this function to return a set


def possible_teams(num_of_people: int, num_of_teams: int) -> set:
    team_size = num_of_people//num_of_teams  # Here I should check for devisibility
    final_result = []  # we can make this a set

    # we can use OOP such as initializing person objects dynamically!!
    ls = [x for x in range(1, num_of_people+1)]  # This line makes this dynamic
    temp = list(itertools.combinations(
        ls, team_size))
    for i in temp:  # each i is a tuple!
        temp_list = [i]  # we can change this to a set

        for j in temp:  # each j is a tuple as well!
            temp1 = np.array(temp_list)
            temp2 = np.array(j)
            if (not np.isin(temp1, temp2).any()):
                temp_list.append(j)  # we can modify this for a set

        # I changed this line, and replaced this with the line below
        final_result.append(temp_list)

    return final_result


all_possible_combination = possible_teams(total_num_of_people, number_of_teams)
# print(possible_teams(6, 2))  # This shows that the above method works!!
# Now we gotta create


# Now lets create a list of possible teams (object)
possible_teams = []
# The variable above shows the format as follows:
# [[u1,u2,u3],[u4,u5,u6],[u1,u5,u6],[u2,u3,u4],..........]
for i in range(0, len(all_possible_combination)):
    for j in range(0, number_of_teams):
        people_list = []
        for k in range(0, total_num_of_people//number_of_teams):
            temp_string = all_possible_combination[i][j][k]
            exec("people_list.append(u%s)" % (temp_string))
            #exec("%s = User(%d, g6, p6)" % (name, 1))
        possible_teams.append(people_list)

# print(possible_teams)  # for debugging

# The variable below stores all the possible teams (without distinguishing among configs)
list_of_team = []  # An important variable!!

for i in range(0, len(possible_teams), number_of_teams):
    #temp = Team(possible_teams[i])
    #temp1 = Team(possible_teams[i+1])
    # list_of_team.append(temp)
    # list_of_team.append(temp1)
    # My mod code starts from here (to make things dynamic)
    t = i
    for j in range(0, number_of_teams):
        list_of_team.append(Team(possible_teams[t]))
        t = t+1

# Check to see that we indeed have added all the teams


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


for config in final_configs:
    print(str(config.performance_score())+"\t" + str(config.fairness_score()))
