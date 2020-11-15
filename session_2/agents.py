"""
File to complete. Contains the agents
"""
import numpy as np
import math


class Agent(object):
    """Agent base class. DO NOT MODIFY THIS CLASS
    """

    def __init__(self, mdp):
        super(Agent, self).__init__()
        # Init with a random policy
        self.policy = np.zeros((4, mdp.env.observation_space.n)) + 0.25
        self.mdp = mdp
        self.discount = 0.9

        # Intialize V or Q depends on your agent
        # self.V = np.zeros(self.mdp.env.observation_space.n)
        # self.Q = np.zeros((4, self.mdp.env.observation_space.n))

    def update(self, state, action, reward):
        # DO NOT MODIFY. This is an example
        pass

    def action(self, state):
        # DO NOT MODIFY. This is an example
        return self.mdp.env.action_space.sample()


class QLearning(Agent):
    def __init__(self, mdp):
        super(QLearning, self).__init__(mdp)

    def update(self, state, action, reward):
        """
        Update Q-table according to previous state (observation), current state, action done and obtained reward.
        :param state: state s(t), before moving according to 'action'
        :param action: action a(t) moving from state s(t) (='state') to s(t+1)
        :param reward: reward received after achieving 'action' from state 'state'
        """
        new_state = self.mdp.observe() # To get the new current state

        # TO IMPLEMENT
        raise NotImplementedError

    def action(self, state):
        """
        Find which action to do given a state.
        :param state: state observed at time t, s(t)
        :return: optimal action a(t) to run
        """
        # TO IMPLEMENT
        raise NotImplementedError


class ValueIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = 0.9

    def optimal_value_function(self):
        """1st step of value iteration algorithm
            Return: State Value V
        """
        # Intialize random V
        V = np.zeros(self.mdp.env.nS)
        
        while True:
            delta = 0
            #loop over the states, not the last one (goal)
            for s in range(self.mdp.env.nS -1):
                Q = np.zeros(self.mdp.env.nA)
                for a in range(self.mdp.env.nA): #all actions
                    for  prob_s, next_state, reward, done in self.mdp.env.P[s][a]:
                        Q[a] += prob_s * (reward + self.gamma * V[next_state]) #calculate value function
                    
                max_value_function_s = np.max(Q)
            
                delta = max(delta, np.abs(max_value_function_s - V[s]))
            
                V[s] = max_value_function_s
        
            if delta < 0.00001: #compare to a threshold theta
                break
        return V

    def optimal_policy_extraction(self, V):
        """2nd step of value iteration algorithm
            Return: the extracted policy
        """
        policy = np.zeros([self.mdp.env.nS, self.mdp.env.nA])
        # TO IMPLEMENT
        #loop over the states, not the last one (goal)
        for s in range(self.mdp.env.nS -1 ):
            Q_sa = np.zeros(self.mdp.env.nA)
            for a in range(self.mdp.env.nA): #for all actions
                for  prob_s, next_state, reward, done in self.mdp.env.P[s][a]:
                    if s==next_state: #I put a high negative reward in order to avoid the same state
                            reward = -1000
                    Q_sa[a] += prob_s * (reward + self.gamma * V[next_state])

            best_action = np.argmax(Q_sa)

            policy[s] = np.eye(self.mdp.env.nA)[best_action]

        return policy

    def value_iteration(self):
        """This is the main function of value iteration algorithm.
            Return:
                final policy
                (optimal) state value function V
        """
        policy = np.random.uniform(0, 1, (self.mdp.env.nS, self.mdp.env.nA))
        V = np.zeros(self.mdp.env.nS)
        # TO IMPLEMENT 
        optimal_v = self.optimal_value_function()
    
        policy = self.optimal_policy_extraction(optimal_v)
        return policy, V


class PolicyIteration:
    def __init__(self, mdp):
        self.mdp = mdp
        self.gamma = 0.9

    def policy_evaluation(self, policy):
        """1st step of policy iteration algorithm
            Return: State Value V
        """
        V = np.zeros(self.mdp.env.nS) # intialize V to 0's

        while True:
            delta = 0
            #loop over the states, not the last one (goal)
            for state in range(self.mdp.env.nS -1): 
                val = 0  
                for action,act_prob in enumerate(policy[state]):  #for all actions 
                    for prob,next_state,reward,done in self.mdp.env.P[state][action]:
                        if state==next_state: #put a high negative reward to avoid the same state
                            reward = -1000
                        val += act_prob * prob * (reward + self.gamma * V[next_state])  
                
               
                delta = max(delta, np.abs(V[state]-val))
                V[state] = val
            if delta < 0.00001:  #compare to threshold theta
                break
        return np.array(V)

    def policy_improvement(self, V, policy):
        """2nd step of policy iteration algorithm
            Return: the improved policy
        """
        #loop over the states, not the last one (goal)
        q = np.zeros(self.mdp.env.nA)
        for s in range(self.mdp.env.nS -1): 
            for a in range(self.mdp.env.nA):
                for prob, next_state, reward, done in self.mdp.env.P[s][a]:
                    if s == next_state: #put a high negative reward to avoid the same state
                        reward = -1000
                    q[a] += prob * (reward + self.gamma * V[next_state])

        return q
    

    def policy_iteration(self):
        """This is the main function of policy iteration algorithm.
            Return:
                final policy
                (optimal) state value function V
        """
        # Start with a random policy
        policy = np.random.uniform(0, 1, (self.mdp.env.nS, self.mdp.env.nA))  # Action in [UP, RIGHT, DOWN, LEFT]

        V = np.zeros(self.mdp.env.nS)
        # To implement: You need to call iteratively step 1 and 2 until convergence
        for i in range(0,500):
            policy, V = self.iterate(policy)
        return policy, V
    
    def iterate(self,policy):
        
        curr_pol_val = self.policy_evaluation(policy)
        policy_stable = True  
        for state in range(self.mdp.env.nS -1):
            chosen_act = np.argmax(policy[state])  
            act_values = self.policy_improvement(curr_pol_val,policy)
                
            best_act = np.argmax(act_values)
            print(chosen_act)
            print("done")
            print(best_act)
            if chosen_act != best_act:
                policy_stable = False 
                policy[state] = np.eye(self.mdp.env.nA)[best_act]  
        
        
        return policy, curr_pol_val
        
        
