import gym
import logging
import os,sys
import numpy as np
import random
import math
from gym.envs.classic_control import cartpole

class StorageSystem:
    action_gap = []
    def __init__(self):
        pass

    def appender(self, qtable):
        self.action_gap.append(np.mean(qtable[4000:6000, 0] - qtable[4000:6000, 1]))

class qlearn(object):
    def __init__(self,
                 num_states=10000,
                 num_actions=2,
                 alpha=0.0025,
                 gamma=0.99,
                 epsilon=1,
                 decay_rate=0.9999):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.state = 0
        self.action = 0
        self.qtable = np.random.uniform(low=-1, high=1, size=(num_states, num_actions))
        #self.qtable = np.zeros(shape = (num_states, num_actions))
        self.Tau = cartpole.CartPoleEnv().tau
        self.gamma = gamma

    def set_initial_state(self, state):
        """
        @summary: Sets the initial state and returns an action
        @param state: The initial state
        @returns: The selected action
        """
        self.state = state
        self.action = self.qtable[state].argsort()[-1]
        return self.action

    def move(self, state_prime, reward):
        """
        @summary: Moves to the given state with given reward and returns action
        @param state_prime: The new state
        @param reward: The reward
        @returns: The selected action
        """
        alpha = self.alpha
        gamma = self.gamma
        state = self.state
        action = self.action
        qtable = self.qtable
        Tau = self.Tau
        SS = StorageSystem()

        choose_random_action = (1 - self.epsilon) <= np.random.uniform(0, 1)

        if choose_random_action:
            action_prime = random.randint(0, self.num_actions - 1)
        else:
            action_prime = self.qtable[state_prime].argsort()[-1]

        self.epsilon = 0.1 if self.epsilon <= 0.1 else (self.epsilon * self.decay_rate)
        #qtable[state, action] = (1 - alpha) * qtable[state, action] + alpha * (reward + gamma * qtable[state_prime, action_prime])
        #math.exp(-Tau/(-0.02/math.log(0.99)))

        delta = reward + pow(gamma,Tau) * qtable[state_prime, action_prime] - qtable[state, action]
        self.qtable[state, action] = qtable[state, action] + alpha * delta

        SS.appender(self.qtable)
        self.state = state_prime
        self.action = action_prime
        return self.action
