import itertools
import numpy as np
import copy as cp
import random as rd


def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1
