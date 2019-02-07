

class MultiArmedBandit:

    def __init__ (self, k, init_weights):
        """
        :param k (int): number of arms
        :param init_weights (string or array of floats): initial values for each arm
            "uniform" : weights are initialized to 1/k
            "zero": weights are initialized to 0
            if a float array of size k is provided, arms are initialized with the given values
        """

        if type(k) == int and k > 0:
            self.k = k
        else:
            raise ValueError

        if type(init_weights) == str:
            if init_weights == "uniform":
                self.init_weights = [1./k] * k
            elif init_weights == "zero":
                self.init_weights = [0.] * k
            else:
                raise ValueError
        elif type(init_weights) == list:
            if len(init_weights) == k and all(isinstance(x, (int, float)) for x in init_weights):
                self.init_weights = init_weights
            else:
                raise ValueError
        else:
            raise NotImplementedError

        self.weights = [i for i in self.init_weights]
        self.N = [0] * self.k
        self.rewards = [0] * self.k

    def pull(self):
        raise NotImplementedError

    def update(self, arm, reward):
        raise NotImplementedError

    def average_reward(self):
        raise NotImplementedError

    def __str__(self):
        """
        Prints a representation of the k-armed bandit
        :return: string
        """
        return "Number of Arms = {}\nInitial Weights = {}\nPulls = {}\nCurrent Weights = {}".format(self.k, self.init_weights, self.N, self.weights)
