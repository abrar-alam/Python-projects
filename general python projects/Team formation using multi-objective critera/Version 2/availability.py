import random as rd


class Availability:
    """This class represents availabilities for each user by building a 24 x 7 = 168 element long binary array.
       This is because each day has 24 hrs, and in a week there are 7 days."""

    def __init__(self):
        # we basically need a 168 elemnt array with each slot being randomly filled with 0 and 1 (binary array)
        self.availability_array = [
            rd.randrange(0, 2, 1) for t in range(1, 169)]
