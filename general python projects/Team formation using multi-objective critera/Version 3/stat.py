import statistics


class Stat:
    def __init__(self):
        raise TypeError("cannot create 'Stat' instances")

    # Mean-max normalization

    @staticmethod
    def min_max_norm(data: list) -> list:
        minimum = min(data)
        maximum = max(data)
        lst = [(x-minimum)/(maximum-minimum) for x in data]
        return lst
