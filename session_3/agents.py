import numpy as np
import random
random.seed(0)
np.random.seed(0)


"""
Contains the definition of the agent that will run in an
environment.
"""

class RandomAgent:
    def __init__(self):
        """Init a new agent.
        """

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        return np.random.randint(0, 10)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        pass

class epsGreedyAgent:
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.mu = {a:[] for a in self.A}
        self.epsilon = 0.1

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        for a in self.A:
            if len(self.mu[a]) == 0:
                return a

        if np.random.uniform(0, 1) < 1 - self.epsilon:
            return np.argmax([np.mean(self.mu[a]) for a in self.A])
        else:
            return np.random.randint(0, 10)

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[action].append(reward)


class BesaAgent():
    # https://hal.archives-ouvertes.fr/hal-01025651v1/document
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        raise NotImplemented

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        raise NotImplemented

class SoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        raise NotImplemented

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        raise NotImplemented

class UCBAgent:
    # https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        raise NotImplemented

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        raise NotImplemented

class ThompsonAgent:
    # https://en.wikipedia.org/wiki/Thompson_sampling
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        raise NotImplemented

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        raise NotImplemented

class KLUCBAgent:
    # See: https://hal.archives-ouvertes.fr/hal-00738209v2
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        raise NotImplemented

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        raise NotImplemented
