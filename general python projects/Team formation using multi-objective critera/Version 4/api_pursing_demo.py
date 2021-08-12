import requests
import json
from availability import Availability
from personality import Personality
from skills import Skills
from gender import Gender
from user import User
import random
import copy as cp
from team import Team
from teams import Teams
url = "https://stg-app.pandos.io/_core_/workspaces/66/users-for-team-formation"

payload = {}
headers = {
    'Authorization': 'Bearer eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJodHRwOi8vbG9jYWxob3N0OjgwODEvYXBpL3YyL2pvaW4vbG9naW4iLCJpYXQiOjE2MjYwOTY1MzIsImV4cCI6MTYyNjExMDkzMiwibmJmIjoxNjI2MDk2NTMyLCJqdGkiOiJYSUxySm9Qalk2ZTNFSDFWIiwic3ViIjozNywicHJ2IjoiZmI5ZDYyNmM1NDQ1ZjRmMGUwYTExNzgyNzA5MzRmZDhjNTNlN2ZlNyJ9.i8eXAg7BJGxnXFhZ88epFHJetzdA99OjtMFlPFZoNG0'
}

response = requests.request("GET", url, headers=headers, data=payload)
response_data = response.json()  # This gives us a dictionary object
# IMPORTANT : If we experience 401 (unauthorized) error then we will experinece type error while indexing 'response_data'
# print(response_data['resource']['users'][0])  # This is a single user
# print(len(response_data['resource']['users'][0]))  # Each user has 5 fields
# Shows how availabilities are being arranged or the format
# print(type(response_data['resource']['users'][0]['availabilities'][0]['day']))
# print(response_data['resource']['users']
# [0]['availabilities'][0]['from_hour'][0:2])
# value = int(response_data['resource']['users']
# [0]['availabilities'][0]['from_hour'][0:2])
# print(value)
# print(response_data['resource']['users'])

"""print(int(response_data['resource']['users']
      [0]['availabilities'][0]['from_hour']))"""
# Now try to build the availability array for the first user


def demo_availability() -> list:
    temp = [0 for t in range(0, 168)]
    for i in range(len(response_data['resource']['users'][0]['availabilities'])):
        value = int(response_data['resource']['users']
                    [0]['availabilities'][i]['from_hour'][0:2])
        if (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Monday':

            temp[0+value] = 1

        elif (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Tuesday':
            temp[24+value] = 1

        elif (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Wednesday':
            temp[48+value] = 1

        elif (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Thursday':
            temp[72+value] = 1

        elif (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Friday':
            temp[72+value] = 1

        elif (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Saturday':
            temp[96+value] = 1

        elif (response_data['resource']['users'][0]['availabilities'][i]['day']) == 'Sunday':
            temp[120+value] = 1
    return temp


# Now work on a function that will return a list of Availability objects. The length of this list is the number of users
def get_availabilities() -> list:
    # A temp array that stores all availability object
    t = []
    for j in range(0, len(response_data['resource']['users'])):
        temp = [0 for t in range(0, 168)]
        for i in range(len(response_data['resource']['users'][j]['availabilities'])):
            value = int(response_data['resource']['users']
                        [j]['availabilities'][i]['from_hour'][0:2])
            if (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Monday':

                temp[0+value] = 1

            elif (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Tuesday':
                temp[24+value] = 1

            elif (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Wednesday':
                temp[48+value] = 1

            elif (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Thursday':
                temp[72+value] = 1

            elif (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Friday':
                temp[72+value] = 1

            elif (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Saturday':
                temp[96+value] = 1

            elif (response_data['resource']['users'][j]['availabilities'][i]['day']) == 'Sunday':
                temp[120+value] = 1
        t.append(Availability(temp))
    return t


# Now we try to capture all the Personalities, and return a list of Personalities


def get_personalities() -> list:
    t = []
    for j in range(0, len(response_data['resource']['users'])):
        # t.append(response_data['resource']['users'][j]["personality"])
        Agr = float(response_data['resource']['users']
                    [j]["personality"][0]['score'])
        Cons = float(response_data['resource']
                     ['users'][j]["personality"][1]["score"])
        ES = float(response_data['resource']['users']
                   [j]["personality"][2]['score'])
        EV = float(response_data['resource']['users']
                   [j]["personality"][3]['score'])
        IN = float(response_data['resource']['users']
                   [j]["personality"][4]['score'])
        t.append(Personality(EV, Cons, IN, Agr, ES))
    return t


# print(response_data['resource']['users'][0]["personality"])  # for debugging
# print("\n")
# tau = get_personalities()
# tau2 = get_availabilities()
# if (len(tau) == len(tau2)):
 #   print("True")


# print(response_data['resource']['users'][0]['personality'])
# Now lets define get skills function
def get_skills() -> list:
    temp = []
    for j in range(0, len(response_data['resource']['users'])):
        t = []
        for i in range(0, len(response_data['resource']['users'][j]['skills'])):
            t.append(response_data['resource']
                     ['users'][j]['skills'][i]['name'])
        temp.append(Skills(t))
    return temp


# Now let's define get gender function. IMPORTANT: I am ignoring the LGBTQ+ cases for now.Besides we are also ingnoring
# the case where gender is not specified
# We are returning a list of gender
def get_gender() -> list:
    temp = []
    for j in range(0, len(response_data['resource']['users'])):
        if(response_data['resource']['users'][j]['gender'] == 'female'):
            temp.append(Gender(1))
        else:
            temp.append(Gender(-1))
    return temp

# Now we grab userID
# We return a list of int uID


def get_id() -> list:
    temp = []
    for j in range(0, len(response_data['resource']['users'])):
        temp.append(response_data['resource']['users'][j]["user_id"])
    return temp


tau = get_availabilities()
tau1 = get_skills()
tau2 = get_personalities()
tau3 = get_gender()
tau4 = get_id()

user_list = [User(tau4[i], tau3[i], tau2[i], tau[i], tau1[i])
             for i in range(len(response_data['resource']['users']))]

for u in user_list:
    print(u)

# IMPORTANT: The number of teams we want to form is specified as a global variable below
"""Important Note:  'total_num_of_people' variable below MUST be divisble by 'number_of_teams' variable. 
   To put simply, we only focus on generating multiple SAME sized teams"""
number_of_teams = 3  # These values can be changed as long as we adhere to the note above
# These values can be changed as long as we adhere to the note above
total_num_of_people = len(user_list)
gender_diversity = True
availability_diversity = True
personality_diversity = False
skills_diversity = False
# Now the method (combinatorics)


def possible_teams(id_list: list, num_of_teams: int) -> list:
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
    num_of_people = len(id_list)
    possible_result = []
    #user_id_list = [t for t in range(1, num_of_people+1)]
    temp2 = cp.deepcopy(id_list)
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


# Python3 helper function to convert a
# list into a tuple
def convert(list):
    return (*list, )


#all_possible_combination = possible_teams([u[i].id for i in range (len(u))], )
# We see that the number of users is 13. So we make it even, that is 12 (temp solution)
print(len(user_list))

all_possible_combination = possible_teams(
    [user_list[i].id for i in range(len(user_list))], number_of_teams)
# print(all_possible_combination)
# print(len(all_possible_combination))
# Now create a dictionary of user objects
u = {user_list[i].id: user_list[i] for i in range(total_num_of_people)}
# print(u)

# Now lets create a list of possible teams (object)
possible_teams = []
# The variable above shows the format as follows:
# [[u1,u2,u3],[u4,u5,u6],[u1,u5,u6],[u2,u3,u4],..........]

for i in range(0, len(all_possible_combination)):
    for j in range(0, 3):
        people_list = []

        for k in range(0, len(user_list)//3):
            temp_string = all_possible_combination[i][j][k]
            exec("people_list.append(u[%d])" % (temp_string))
            #exec("%s = User(%d, g6, p6)" % (name, 1))
        possible_teams.append(people_list)

# print(len(possible_teams))
# print(possible_teams)
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
# for team in list_of_team:  # This works fine!!
 #   print(team.toString())

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


# Now we store performance and fairness scores associated with each configuration
raw_objective_values = []  # An important variable!

# the method below builds 'raw_objective_values'


def build_objective_val(x: list):
    for config in list_of_configurations:
        x.append((config.performance_score(), config.fairness_score()))


# Now we build 'raw objective values'
build_objective_val(raw_objective_values)  # Very important!!

print(raw_objective_values)  # Works fine!

print(len(raw_objective_values))  # works fine!
