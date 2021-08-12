from gender import Gender
from personality import Personality
from availability import Availability
from skills import Skills


class User:
    """This class represents a single user. A user can have multiple 
        criteria such as availablity, personality trait, gender, skill sets, goals, leadership
        styles, etc. """
    incremental_id = 0
    # Constrcutor

    def __init__(self, id, gender: Gender, personality: Personality, availability: Availability):
        self.id = id
        self.gender = gender
        self.personality = personality
        self.availability = availability
        User.incremental_id += 1

    # Another constructor, useful in generating random users
    def __init__(self):
        User.incremental_id += 1
        self.id = User.incremental_id
        self.gender = Gender()
        self.personality = Personality()
        self.availability = Availability()
        self.skills = Skills()

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
