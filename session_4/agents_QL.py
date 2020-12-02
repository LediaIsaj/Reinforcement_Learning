import numpy as np
import environment 
from sklearn.linear_model import LinearRegression

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
        self.pos_space = np.linspace(-1.2, 0.6, 12)
        self.vel_space = np.linspace(-0.07, 0.07, 20)
        self.action_space = [0, 1, 2]
        self.alpha = 0.1
        self.gamma = 0.9
        self.eps = 1
        self.model = LinearRegression()
            
        self.variables = [] #position,velocity,action
        self.qvalues = [] #after update
        
        states = []
        for pos in range(21):
            for vel in range(21):
                states.append((pos, vel))

        self.Q = {}
        for state in states:
            for action in self.action_space:
                self.Q[state, action] = 0
    
    def get_state(self,observation):
        pos, vel =  observation
        pos_bin = int(np.digitize(pos, self.pos_space))
        vel_bin = int(np.digitize(vel, self.vel_space))

        return (pos_bin, vel_bin)

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
        #if first trial or less then epsilon(explore)--> random choice, else be greedy
        if (len(self.Q) == 0 or np.random.random() < self.eps):
            action = np.random.choice([0,1,2])
        else:
            obs = self.get_state(state)
            list_q = []
            for a in self.action_space:
                list_q.append(self.Q[obs,a])
            values = np.array([list_q])
            action = np.argmax(values)  
        return action
            

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
        obs = self.get_state(state)
        new_obs = self.get_state(new_state)
        list_q = []
        for a in self.action_space:
            list_q.append(self.Q[obs,a])
        values = np.array([list_q])
        new_action = np.argmax(values)  
        self.Q[obs, action] = self.Q[obs, action] + self.alpha*(reward + self.gamma*self.Q[new_obs, new_action] - self.Q[obs, action])
        

    def q(self, state, action):
        """Final Q function. It will be used for visualization purposes.
        Parameters:
            state: vector in R^2
            action: scalar (0, 1 or 2)
        Return:
            Value (scalar) of Q(state, action)
        """
        obs = self.get_state(state)
        return self.Q[obs, action]




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
