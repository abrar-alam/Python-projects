from personality import Personality
from gender import Gender
from user import User
from availability import Availability
from abc import ABC, abstractmethod
import copy as cp
import statistics as st


class AbstractTeam(ABC):
    """This abstract class provides an API for possible concrete teams which can be formed based on both 'similarity' and 'diversity' criteria"""

    def __init__(self, userslist: list):
        self.userslist = cp.deepcopy(userslist)

    @abstractmethod
    def gender_performance(self) -> float:
        pass

    @abstractmethod
    def personality_performance(self) -> float:
        pass

    @abstractmethod
    def skills_performance(self) -> float:
        pass

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
        return (total/168.0)

    @abstractmethod
    def total_team_performance(self) -> float:
        pass

    @abstractmethod
    def toString(self) -> str:
        pass

# A helper method to calculate a team's availability performance


def multiply_arrays(a1, a2):
    x = []
    for i in range(len(a1)):
        x.append((a1[i])*(a2[i]))
    return x
