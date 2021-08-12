import random as rd


class Gender:
    """This class represents a user's gender. This can be binary or non-binary
       However, the non-binary feartures is currently under experimentations. Thus, this only
       shows binary genders, such as, male and female"""

    # Constrcutor
    """ The value is a binary value. -1 for male, and 1 for female"""
    """we, however, plan to represent gender by a string, this will be done after the experimentation"""
    # Use pronouns such as 'HE', 'SHE, and 'THEY'. # Ask people whats pronouns they pre4fer. Then assign appropriate
    # pronouns to 'Gender' object

    def __init__(self, value):
        self.value = value

    # Overloaded constructor to help generating random gender
    def __init__(self):
        # First we generate a random number (either 1 or -1)
        t = rd.randrange(-1, 2, 2)
        self.value = t

    def __str__(self) -> str:
        if self.value == -1:
            return "Male"
        return "Female"
