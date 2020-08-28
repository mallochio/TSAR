# Watkins' Q-Learning with discretized spaces for cartpole simulation #

import gym
import numpy as np
import random
import pandas as pd
import qlearn as ql
from gym import wrappers
from altair import *
from matplotlib import pyplot as plt
plt.style.use('ggplot')

def plotAltair(data):
     Chart(data).mark_area().encode(
          X('Episode:T', axis=Axis(format = "%d", labelAngle=0)),
          y='trial1'
     ).savechart('./pict.json', filetype = 'json')

def smoothListGaussian(list, degree = 20):
     window      = degree * 2 - 1
     weight      = np.array([1.0] * window)
     weightGauss = []

     for i in range(window):
         i     = i - degree + 1
         frac  = i/float(window)
         gauss = 1/(np.exp((4.0 * (frac)) ** 2.0))
         weightGauss.append(gauss)

     weight   = np.array(weightGauss) * weight
     smoothed = [0.0] * (len(list) - window)

     for i in range(len(smoothed)):
         smoothed[i] = sum(np.array(list[i:i + window]) * weight)/sum(weight)

     return smoothed

def plotster(arr):
     arr = smoothListGaussian(arr)
     x = np.linspace(0, len(arr), len(arr))
     #pd.DataFrame(data = arr[0:, 0:], index = x, columns = np.asarray([0, 1]))

     fig, ax = plt.subplots()
     m, b = np.polyfit(x, arr, 1)
     print "learning slope: ", m
     ax.plot(arr)
     ax.plot(x, m*x + b, '-')
     plt.show()

# Find best 100 episode score
def findMax(arr):
     n = len(arr)
     k = 100
     if k > n:
          print "need more episodes"
          return -1
     else:
          sum = np.sum(arr)
          max_sum = sum
          max_end = k-1

          for i in xrange(k, n-1):
               sum += arr[i] - arr[i-k]
               if sum > max_sum:
                    max_sum = sum
                    max_end = i

     return (np.mean(arr[max_end - k + 1 : max_end + 1]))


def cart_pole_with_qlearning(alpha, gamma, epsilon, decay_rate):
     random.seed(0)
     experiment_filename = './cartpole-experiment-1'
     env                 = wrappers.Monitor(gym.make('CartPole-v1'), experiment_filename, force = True)
     number_of_features  = env.observation_space.shape[0]
     steps_to_completion = []
     q_mean = []
     cart_position_bins = pd.cut([-2.4, 2.4], bins=10, retbins=True)[1][1:-1]
     pole_angle_bins    = pd.cut([-0.24, 0.24], bins=10, retbins=True)[1][1:-1]
     cart_velocity_bins = pd.cut([-3.5, 3.5], bins=10, retbins=True)[1][1:-1]
     angle_rate_bins    = pd.cut([-3.2, 3.2], bins=10, retbins=True)[1][1:-1]

     def build_state(features):
          return int("".join(map(lambda feature: str(int(feature)), features)))

     def to_bin(value, bins):
          return np.digitize(x=[value], bins=bins)[0]

     num_states               = 10 ** number_of_features
     num_actions              = env.action_space.n
     learner = ql.qlearn(num_states, num_actions, alpha, gamma, epsilon, decay_rate)

     ## For each episode do
     for episode in xrange(500):
          observation = env.reset()
          cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
          temp = learner.qtable.copy()
          temp = [sub_list[0] for sub_list in temp]

          q_mean.append(np.mean(temp))
          state = build_state([to_bin(cart_position, cart_position_bins),
                               to_bin(cart_velocity, cart_velocity_bins),
                               to_bin(pole_angle, pole_angle_bins),
                               to_bin(angle_rate_of_change, angle_rate_bins)])
          action = learner.set_initial_state(state)
          stepcount = 0

          ## For each step do
          while True: 
               stepcount += 1
               #env.render()
               observation, reward, done, info = env.step(action)
               cart_position, cart_velocity, pole_angle, angle_rate_of_change = observation
               state_prime = build_state([to_bin(cart_position, cart_position_bins),
                                          to_bin(cart_velocity, cart_velocity_bins),
                                          to_bin(pole_angle, pole_angle_bins),
                                          to_bin(angle_rate_of_change, angle_rate_bins)])

               if done:
                    steps_to_completion.append(stepcount)
                    reward = -200
               action = learner.move(state_prime, reward)
               if done:
                    break

     [sub_list[0] for sub_list in learner.qtable]
     freqMult  = 1
     steps_to_completion = [i/(500.0 * freqMult) for i in steps_to_completion] # Normalization
     lossmax   = findMax(steps_to_completion)
     meanscore = np.mean(steps_to_completion)
     print "max: ", lossmax
     print "mean: ", meanscore

     #plotster(steps_to_completion)
     env.close()
     #gym.upload( experiment_filename, api_key = 'sk_MD4yPVVeRbCwnaorTY4LuQ')
     return steps_to_completion
     #return lossy
     #return lossmax

if __name__ == '__main__':
#def main(job_id, params):
     #start = timer()
     random.seed(0)

     steps = []

     trial = []
     #print 'Anything printed here will end up in the output directory for job #%d' % job_id
     #print params
     for trialNum in xrange(0,3):
          steps_to_completion = cart_pole_with_qlearning(0.758703 , 0.99, 0.624607, 0.7)
          for j in steps_to_completion:
               trial.append(trialNum)
          steps.append([trial, steps_to_completion])

     print np.shape((np.linspace(0,499,500)))
     print np.shape((steps[:][0]))

#    data = pd.DataFrame({'Episode': list(np.linspace(0,499,500)), 'trial' : list(steps[:][0]), 'steps' : list(steps[:][1]) })
     #plotAltair(data)

     #cart_pole_with_qlearning(0.000023, 0.989990, 0.967411, 0.989436)
     #return cart_pole_with_qlearning(params['alpha'], params['gamma'], params['epsilon'], params['decay_rate']) # for spearmint




