from decimal import DivisionByZero
from personality import Personality
from gender import Gender
from user import User
from similarityfocusedteam import SimilarityFocusedTeam
from diversityfocusedteam import DiversityFocusedTeam
import copy as cp
import statistics as st


class Teams:
    """This class represents an a single collection of individual teams (see Team class for details)
       To be specific, this class represnts a collection of Team objects. The reason we make a seperate class
       such as this is because we need methods for evaluating performance and fairness functions by taking 
       all teams into consideration, not just a single team
       or to put simply, this class represents a 'single' possible team configuration (consisting of multiple teams)"""

    # constructor
    def __init__(self, id: int, list_of_teams):
        self.id = id  # This id starts from 1
        self.list_of_teams = list_of_teams

    # z score normalizer (helper method)
    def z_norm(self, data: list) -> list:
        #minimum = min(data)
        #maximum = max(data)
        mean_val = st.mean(data)
        stddev = st.stdev(data)
        lst = [(x-mean_val)/(stddev) for x in data]
        return lst

    # get overall performance function value (gender). We don't normalize now
    def total_gender_performance(self) -> float:
        # First we get a list of each team's performance(gender) within this teams
        tmp = [team.gender_performance() for team in self.list_of_teams]
        # Each element of 'tmp' above  represents # gender performance of each team
        return sum(tmp)

    # get the overall performance function value (based on big 5 traits).
    def total_personality_performance(self) -> float:
        tmp = [team.personality_performance() for team in self.list_of_teams]

        # Printing the first total personality performance:#Added this for debuggin
        # print(tmp)  # added this for easy debug
        total = sum(tmp)
        # now we normalize(don't need this, we will do this at final stage, when we combine all types of performances such as personality, gender, etc.)
        #tmp = self.min_max_norm(tmp)
        # print(tmp)  # added this for debugging
        # before returning total we make sure to make it in-between 0 and 1
        # return (total/len(tmp))Commented for debugging
        return total

    # Now the total performance function based on the availability performances of each constituent teams
    # We normalize the value based on how many constituent teams we have in a single configuration
    def total_availability_performance(self) -> float:
        tmp = [team.availability_performance() for team in self.list_of_teams]
        total = sum(tmp)
        return (total/len(self.list_of_teams))

    # Now we get the total performance score based on all possible criteria ( we only considered gender and personality here).
    # However, other criteria can easily be added by slight modifications.
    # This method is very likely to need some normalization to prevent biases. Howver, we worry about this later

    def performance_score(self) -> float:
        return (self.total_gender_performance() + self.total_personality_performance() + self.total_availability_performance())

    # Now we define fairness, that is, the SD of performance scores of all teams within a 'single' configuration
    # the lower this value, the fairer our configuration is!!
    def fairness_score(self) -> float:
        tmp = [team.total_team_performance() for team in self.list_of_teams]
        # Now we compute the performance disparity by taking SD
        val = st.stdev(tmp)
        return val

    # This method might help to debug
    def toString(self) -> str:
        temp = ""
        for team in self.list_of_teams:
            temp = temp + team.toString() + "; "
        return temp
