from gender import Gender
from personality import Personality


class User:
    """This class represents a single user. A user can have multiple 
        criteria such as availablity, personality trait, gender, skill sets, goals, leadership
        styles, etc. """

    # Constrcutor
    def __init__(self, id, gender: Gender, personality: Personality):  # will add availability later
        self.id = id
        self.gender = gender
        self.personality = personality
        #self.availablity = availability

    # User representation
    # to be implemented a bit later
    def __str__(self) -> str:
        string = "User "+str(self.id)+"\n"+"\t" +\
            "gender: " + self.gender.__str__()+"\n"+"\t" +\
            "Personality traits: \n"+self.personality.__str__()

        return string

    # This method has been added to help debugging
    def toString(self) -> str:
        return str(self.id)
