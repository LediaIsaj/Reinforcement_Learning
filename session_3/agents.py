import numpy as np
import random
import math
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
            return np.random.randint(0, 10) # random exploration part

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
        self.mu = {a:[] for a in self.A}
    
    def choose_2_arms(self,a1,a2):
        
        actions = [a1,a2]
        
        na1 = len(self.mu[a1])
        if (na1 == 0):
            return a1
        na2 = len(self.mu[a2])
        if (na2 == 0):
            return a2
        
        
        
        if na1 == na2: #if the arms have been chosen the same amout of times, choose the empirical best mu
            return actions[np.argmax([np.mean(self.mu[a1]),np.mean(self.mu[a2])])]
        
        else:
            if na1 > na2: # if arm1 has been chosen more times, ramdomly sample from rewards of a1 (na2 times)
                new_list = random.sample(self.mu[a1], na2)
                return actions[np.argmax([np.mean(new_list),np.mean(self.mu[a2])])]
            
            else:
                new_list = random.sample(self.mu[a2], na1) # if arm2 has been chosen more times, ramdomly sample from rewards of a2 (na1 times)
                return actions[np.argmax([np.mean(new_list),np.mean(self.mu[a1])])]
    
    def chunkIt(self,seq, num):
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        return out
    
    def choose(self, arms = [0,1,2,3,4,5,6,7,8,9]):
        """Acts in the environment.

        returns the chosen action.
        """
        if len(arms) == 1: # base condition
            return arms[0]
        
        else:
            shuffled_arms = random.sample(arms, len(arms))   # shuffle to randomize
            chunks = self.chunkIt(shuffled_arms,2) # divide in 2 parts
            return self.choose_2_arms(self.choose(chunks[0]),self.choose(chunks[1]))
          

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[action].append(reward)

class SoftmaxAgent:
    # https://www.cs.mcgill.ca/~vkules/bandits.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.mu = {a:[] for a in self.A}
        self.tau = 0.1
        uniform_prob = np.exp(0.1 / self.tau)
        self.probs = [uniform_prob for a in self.A] # initialize all arms with the same probability

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        sum = np.sum(self.probs)
        probability = (x/sum for x in self.probs)

        action=random.choices(self.A, weights=probability, k=1)
        return action[0]

    def update(self, action, reward):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[action].append(reward)
        new_mean = np.mean(self.mu[action])
        #adapt the probability based on the new mean of rewards
        prob = np.exp(new_mean / self.tau)
        self.probs[action]=prob

class UCBAgent:
    # https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf
    def __init__(self):
        self.A = [0,1,2,3,4,5,6,7,8,9]
        self.mu = {a:[] for a in self.A}
        self.initialized = 0
        self.nj = 0

    def choose(self):
        """Acts in the environment.

        returns the chosen action.
        """
        self.nj +=1
        if (self.initialized < 10):
            chosen = self.A[self.initialized]
            self.initialized +=1
            return chosen
        else:
            return np.argmax([np.mean(self.mu[a]) + math.sqrt(2* math.log(len(self.mu[a]))/self.nj) for a in self.A])

    def update(self, a, r):
        """Receive a reward for performing given action on
        given observation.

        This is where your agent can learn.
        """
        self.mu[a].append(r)

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
