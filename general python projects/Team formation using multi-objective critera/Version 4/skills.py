import random as rd


class Skills:
    """This class represents a set of all chosen arbitrary skills of a single user. For ease of performance measurements
        we implement this set as a list"""

    def __init__(self, list_of_skills: list):
        self.skillset = list_of_skills

    def __str__(self) -> str:
        string = ""
        for skill in self.skillset:
            string = string + skill+"\t"

        return string

# The remaining sections of this file is defined to facilliate random skillsets generation for random user only
# So they will be of no use when the software will be implemented on the server


# Predefined skill sets to assign a random user
s1 = ['Python', 'ML', 'Excel']
s2 = ['NLP', 'eng. Drawing', 'Python']
s3 = ['Excel', 'ML', 'c++']
s4 = ['Word', 'Excel', 'Python']
ls = [s1, s2, s3, s4]

# A helper method


def generate_random_skills() -> list:
    return (ls[rd.randrange(0, 4, 1)])
