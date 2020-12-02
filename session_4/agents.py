import numpy as np
import environment 
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

from collections import deque
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
        self.memory  = deque(maxlen=2000)
        self.action_space = [0,1,2]
        self.states = 2
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.tau = 0.05
        self.model = self.create_model()
        self.target_model = self.create_model()
        
    def create_model(self):
        model   = Sequential()
        model.add(Dense(24, input_dim=self.states, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(len(self.action_space)))
        model.compile(loss="mean_squared_error",
        optimizer=Adam(lr=self.learning_rate))
        return model
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])
        return
    
    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return
        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(np.array([state]))
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(np.array([state]))[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(np.array([state]), target, epochs=1, verbose=0)
        return
            
    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i]
        self.target_model.set_weights(target_weights)
        return

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
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        return np.argmax(self.model.predict(np.array([state]))[0])


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
        print("State",state)
        print("New state",new_state)
        self.remember(state, action, reward, new_state, terminal)
        self.replay()
        self.target_train()

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        return self.model.predict(np.array([state]))[action]




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
