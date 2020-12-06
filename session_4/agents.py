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
        #Initialize all the hyperparameters and create the models
        
        self.memory  = []
        self.action_space = [0,1,2] #push left, no push, push right
        self.states = 2 #number of states
        self.gamma = 0.9 #discount factor
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.01
        self.pred = self.create_model() # pred model is to do predictions for what action to take
        self.track = self.create_model() # track model tracks the desired actions
        
    def create_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.states, activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(24, activation="relu"))
        model.add(Dense(len(self.action_space)))
        model.compile(loss="mean_squared_error",
        optimizer=Adam(lr=self.learning_rate))
        return model
    
    def memorize(self, state, action, reward, new_state, done):
        #this is where we store in memory, so we can take adventage of past experience
        step = [state, action, reward, new_state, done]
        self.memory.append(step)
        return
    
    def replay(self):
        batch = 32
        if len(self.memory) < batch: 
            return
        
        steps = random.sample(self.memory, batch) #sample from memory
        
        for step in steps:
            state, action, reward, new_state, done = step
            track = self.track.predict(np.array([state]))
            if done:
                track[0][action] = reward
            else:
                # Calculate Q as the sum of the current reward and expected future rewards*gamma
                track[0][action] = reward +  max(self.track.predict(np.array([state]))[0]) * self.gamma
            self.pred.fit(np.array([state]), track, epochs=1, verbose=0)
            
        return
            
    def track_update(self):
        #update the weights of track model
        pred_weights = self.pred.get_weights()
        track_weights = self.track.get_weights()
        n = len(track_weights)
        for i in range(n):
            track_weights[i] = pred_weights[i]
        self.pred.set_weights(track_weights)


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
            return np.random.choice(self.action_space) #exploration, choose randomly from action space
        else:
            return np.argmax(self.pred.predict(np.array([state]))[0])


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

        self.memorize(state, action, reward, new_state, terminal)
        self.replay()
        self.track_update()

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        return self.pred.predict(np.array([state]))[action]




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
