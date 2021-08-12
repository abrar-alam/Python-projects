class Personality:
    """This class represents a user's big 5 personality traits. This spectrum consists of 5
        traits, such as, extraversion, conscientiousness, openness, agreeableness, and neuroticism.
        Each trait is represented by a perventile value
        """

    def __init__(self, trait1, trait2, trait3, trait4, trait5):
        self.extraversion = trait1
        self.conscience = trait2
        self.openness = trait3
        self.agreeableness = trait4
        self.neuroticism = trait5

    def __str__(self) -> str:
        string = "Extraversion: " + str(self.extraversion)\
            + "\n"+"Conscientiousness: " + str(self.conscience) + "\n"\
            + "Openness: " + str(self.openness)+"\n"\
            + "Agreeableness: " + str(self.agreeableness)+"\n"\
            + "Neuroticism: "+str(self.neuroticism) + "\n"
        return string
