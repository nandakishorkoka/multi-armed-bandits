from MultiArmedBandit import MultiArmedBandit
import random


class EpsilonGreedyBandit(MultiArmedBandit):

    def __init__ (self, k, init_weights, e):
        """
        :param k (int): number of arms
        :param init_weights (string or array of floats): initial values for each arm
            "uniform" : weights are initialized to 1/k
            "zero": weights are initialized to 0
            if a float array of size k is provided, arms are initialized with the given values
        :param e (float): epsilon
        """

        super().__init__(k, init_weights)
        if type(e) == float and 0 <= e <= 1:
            self.e = e

    def __str__(self):
        """
        string representation of the e-greedy bandit
        :return : str
        """
        return super().__str__() + "\nepsilon = {}".format(self.e) + "\nRewards = {}".format(self.rewards)

    def pull(self):
        """
        pulls a arm at random with probability e
        pulls the best arm with probability 1-e
        :return: int (index of the arm)
        """
        t = random.uniform(0, 1)
        if t <= self.e:
            arm = random.choice(range(0, self.k))
            self.N[arm] += 1
        else:
            max_weight = max(self.weights)
            arms = [i for i in range(self.k) if self.weights[i] == max_weight]
            arm = random.choice(arms)
            self.N[arm] += 1
        return arm

    def update(self, arm, reward):
        """
        update weight of the arm given the rewrd
        :param arm: int (index of the arm)
        :param reward: int (numerical reward)
        :return: None
        """
        self.rewards[arm] += reward
        self.weights[arm] += (1 / self.N[arm]) * (reward - self.weights[arm])

    def average_reward(self):
        if sum(self.rewards) == 0:
            return 0
        return sum(self.rewards) / sum(self.N)
