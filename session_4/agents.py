import numpy as np
import environment

"""
Contains the definition of the agent that will run in an
environment.
"""

class Q_Learning_Function_Approximation:
    """ Q-Learning with Function Approximation
    """

    def __init__(self):
        """Init a new agent.
        """
        pass

    def act(self, state):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        return np.random.choice([0, 1, 2])

    def update(self, state, action, reward, new_state, terminal):
        """Receive a reward for performing given action.

        This is where your agent can learn. (Build model to approximate Q(s, a))
        Parameters:
            state: current state
            action: action done in state
            reward: reward received after doing action in state
            new_state: next state
            terminal: boolean if new_state is a terminal state or not
        """
        pass

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        return np.random.uniform(0, 1)



class Double_Q_Learning:
    """ Q-Learning with Function Approximation
    """

    def __init__(self):
        """Init a new agent.
        """
        pass

    def act(self, state):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
        Range observation (tuple):
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        return [np.random.choice([0, 1, 2])]

    def update(self, state, action, reward, new_state, terminal):
        """Receive a reward for performing given action.

        This is where your agent can learn. (Build model to approximate Q(s, a))
        Parameters:
            state: current state
            action: action done in state
            reward: reward received after doing action in state
            new_state: next state
            terminal: boolean if new_state is a terminal state or not
        """
        pass

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        return np.random.uniform(0, 1)
