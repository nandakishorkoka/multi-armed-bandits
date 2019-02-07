from Environment import Environment
import random


class StaticEnvironment(Environment):

    def __init__(self, k, static_rewards=None):
        """
        Initialize a static environment
        :param k: int (number of actions
        :param static_rewards: list of size k (rewards )
        """
        super().__init__(k)
        if static_rewards and type(static_rewards) == list and len(static_rewards) == k:
            self.rewards = static_rewards
        else:
            self.rewards = [random.random() for i in range(k)]

    def __str__(self):
        return super().__str__() + "\nRewards = {}".format(self.rewards)

    def evaluate(self, arm):
        """
        Evaluate an action and return reward
        :param arm: int (index of the arm
        :return: float (reward)
        """
        if 0 <= arm < self.k:
            return self.rewards[arm]
        else:
            raise ValueError
