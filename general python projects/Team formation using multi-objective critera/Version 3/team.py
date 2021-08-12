#from personality import Personality
#from gender import Gender
#from user import User
#from availability import Availability
#from abc import ABC, abstractmethod
import copy as cp
import statistics as st
from collections import Counter


class Team:
    """Add a proper documentation here"""

    def __init__(self, userslist: list, gender_diversity: bool, personality_diversity: bool, skills_diversity: bool, availability_diversity: bool):
        self.userslist = cp.deepcopy(userslist)
        self.gender_diversity = gender_diversity
        self.personality_diversity = personality_diversity
        self.skills_diversity = skills_diversity
        self.availability_diversity = availability_diversity

    def gender_performance(self) -> float:
        num_of_males = 0
        num_of_females = 0
        for person in self.userslist:
            if (person.gender.value == -1):  # male case
                num_of_males = num_of_males + person.gender.value
            if (person.gender.value == 1):  # female case
                num_of_females = num_of_females + person.gender.value

        # Now we must take the abs of num_of_males (because so far we have negative values, see Gender class for details)
        num_of_males = abs(num_of_males)
        if (self.gender_diversity):
            return (-1*abs(num_of_males - num_of_females))
        return (1*abs(num_of_males - num_of_females))

    def personality_performance(self) -> float:
        # First we take SD of each trait within the team
        # This lists RAW extraversion scores for all users within the team
        extrav_list = [
            user.personality.extraversion for user in self.userslist]
        # similarly we create other traits' lists below
        # For conscience
        conscience_list = [
            user.personality.conscience for user in self.userslist]
        openness_list = [user.personality.openness for user in self.userslist]
        agreeableness_list = [
            user.personality.agreeableness for user in self.userslist]
        neuroticism_list = [
            user.personality.neuroticism for user in self.userslist]

        # Now we take SD for each traits
        extrv_SD = st.stdev(extrav_list)
        conscience_SD = st.stdev(conscience_list)
        openness_SD = st.stdev(openness_list)
        agreeableness_SD = st.stdev(agreeableness_list)
        neuroticism_SD = st.stdev(neuroticism_list)

        # Now we take the average of all trait's SDs, which we argue to be RAW personality performance score for a team
        if (self.personality_diversity):
            return (st.mean([extrv_SD, conscience_SD, openness_SD, agreeableness_SD, neuroticism_SD]))
        return (-1*st.mean([extrv_SD, conscience_SD, openness_SD, agreeableness_SD, neuroticism_SD]))

    def skills_performance(self) -> float:

        temp = []
        t = 0
        for user in self.userslist:
            temp = temp + user.skills.skillset
        counter = Counter(temp)
        if (len(counter) == 0):
            return t
        for key in counter:
            t += (counter[key]-1)
        if (self.skills_diversity):
            return -1*t
        return t

    """Our only concrete method below, because this method remains the same regardless of a diversity focused team or 
       a similarity focused team"""

    def availability_performance(self) -> float:
        # first we calculate how many hours are in common among all the users in 'userslist'
        # Then we divide that number by 168 to noramzlize it
        temp = [1 for t in range(1, 169)]
        for user in self.userslist:
            temp = multiply_arrays(temp, user.availability.availability_array)

        total = 0

        for i in range(0, len(temp)):
            total += temp[i]

        # Now normalize
        if (self.availability_diversity):
            # pay attention to how we could normalize this negatrive number.(imp)
            return (-1*(total/168.0))
        return (total/168.0)

    def total_team_performance(self) -> float:
        return (self.gender_performance() + self.personality_performance()+self.availability_performance()+self.skills_performance())

    def toString(self) -> str:
        temp = ""
        for i in range(0, len(self.userslist)):
            temp = temp + self.userslist[i].toString()+" "
        return temp

# A helper method to calculate a team's availability performance


def multiply_arrays(a1, a2):
    x = []
    for i in range(len(a1)):
        x.append((a1[i])*(a2[i]))
    return x
