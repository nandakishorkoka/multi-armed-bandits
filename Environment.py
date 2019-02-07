

class Environment:

    def __init__(self, k):
        """
        :param k: int (number of actions)
        """
        if type(k) == int and k > 0:
            self.k = k
        else:
            raise ValueError

    def __str__(self):
        return "Num action = {}".format(self.k)

    def evaluate(self, arm):
        raise NotImplementedError

