import numpy as np
import gym
np.random.seed(0)
"""
This file contains the definition of the environment
in which the agents are run.
"""

gym.logger.set_level(40)

class FA_Environment:
    # List of the possible actions by the agents

    def __init__(self):
        """Instanciate a new environement in its initial state.
        """
        self.env = gym.make("MountainCar-v0")
        self.state = self.env.reset()
        self.env.seed(np.random.randint(1, 1000))
        self.nb_step = 0

    def sample_action(self):
        return self.env.action_space.sample()

    def reset(self):
        self.nb_step = 0
        self.state = self.env.reset()

    def render(self):
        self.env.render()

    def observe(self):
        """Returns the current observation that the agent can make
        of the environment, if applicable.
        """
        return self.state

    def act(self, action):
        """Perform given action by the agent on the environment,
        and returns a reward.
        """
        self.state, reward, done, info = self.env.step(action)
        self.nb_step += 1

        return (reward, done)
