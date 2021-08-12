import random as rd


class Availability:
    """This class represents availabilities for each user by building a 24 x 7 = 168 element long binary array.
       This is because each day has 24 hrs, and in a week there are 7 days."""

    # availability_list is the 168 element long binary array
    def __init__(self, availability_list: list):
        # we basically need a 168 elemnt array with each slot being randomly filled with 0 and 1 (binary array)
        self.availability_array = availability_list
