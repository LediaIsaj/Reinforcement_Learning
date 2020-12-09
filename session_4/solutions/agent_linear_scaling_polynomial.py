import numpy as np
import environment
import gym
import sklearn
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

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
        self.gamma = 1.0
        self.scaler = StandardScaler()
        self.feature_generation = PolynomialFeatures(2)

        init_samples = np.array([[np.random.uniform(-1.2, 0.6), np.random.uniform(-0.07, 0.07)] for _ in range(10000)])
        init_samples_poly = self.feature_generation.fit_transform(init_samples)
        self.scaler.fit(init_samples_poly)

        self.models = []
        for _ in range(3):
            model = SGDRegressor(learning_rate="constant")
            model.partial_fit([self.preprocessing([np.random.uniform(-0.6, -0.4), 0])], [-1]) # dirty !
            self.models.append(model)

    def preprocessing(self, state):
        """
        Returns the featurized representation for a state.
        """
        return self.scaler.transform(self.feature_generation.transform([state]))[0]

    def act(self, state):
        """Acts given an observation of the environment.

        Takes as argument an observation of the current state, and
        Range observation (tuple):
        returns the chosen action.
        See environment documentation: https://github.com/openai/gym/wiki/MountainCar-v0
        Possible action: [0, 1, 2]
            - position: [-1.2, 0.6]
            - velocity: [-0.07, 0.07]
        """
        q_values_new_state = [self.q(state, a) for a in [0, 1, 2]]
        best_action = np.argmax(q_values_new_state)
        if np.random.uniform(0, 1) < 0.9:
            return best_action
        else:
            l_actions = [0, 1, 2]
            l_actions.remove(best_action)
            return np.random.choice(l_actions)

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
        q_values_new_state = [self.q(new_state, a) for a in [0, 1, 2]]
        if new_state[0] >= 0.5: # target reached
            target = reward
        else:
            target = reward + self.gamma * np.max(q_values_new_state)
        self.models[action].partial_fit([self.preprocessing(state)], [target])


    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Return:
            Value (scalar) of Q(state, action)
        """
        return self.models[action].predict([self.preprocessing(state)])[0]
