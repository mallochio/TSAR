import numpy as np
import gym
import random
from gym import wrappers
import tensorflow as tf
from matplotlib import pyplot as plt
import pandas as pd
plt.style.use('ggplot')
from matplotlib import rcParams
rcParams['figure.figsize'] = 15, 10


def smoothListGaussian(list, degree=20):
    window = degree * 2 - 1
    weight = np.array([1.0] * window)
    weightGauss = []

    for i in range(window):
        i = i - degree + 1
        frac = i / float(window)
        gauss = 1 / (np.exp((4.0 * (frac)) ** 2.0))
        weightGauss.append(gauss)

    weight = np.array(weightGauss) * weight
    smoothed = [0.0] * (len(list) - window)

    for i in range(len(smoothed)):
        smoothed[i] = sum(np.array(list[i:i + window]) * weight) / sum(weight)

    return smoothed


def plotster(arr):
    # arr = smoothListGaussian(arr)
    x = np.linspace(0, len(arr), len(arr))
    # pd.DataFrame(data = arr[0:, 0:], index = x, columns = np.asarray([0, 1]))

    fig, ax = plt.subplots()
    # m, b = np.polyfit(x, arr, 1)
    # print "learning slope: ", m
    ax.plot(arr)
    # ax.plot(x, m*x + b, '-')
    plt.show()

def medfilt (x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros ((len (x), k), dtype=x.dtype)
    y[:,k2] = x
    for i in range (k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i+1)] = x[j:]
        y[-j:, -(i+1)] = x[-1]
    return np.median (y, axis=1)

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

    #return 1/(np.mean(arr[max_end - k + 1 : max_end + 1]))
    print "TopScore", (np.mean(arr[max_end - k + 1 : max_end + 1]))


class DQN:
    REPLAY_MEMORY_SIZE = 1000000
    RANDOM_ACTION_DECAY = 0.9999
    MIN_RANDOM_ACTION_PROB = 0.1
    HIDDEN1_SIZE = 64  # 32
    HIDDEN2_SIZE = 32  # 64
    HIDDEN3_SIZE = 64  # 512
    MAX_STEPS = 5000000
    LEARNING_RATE = 0.0025
    MINIBATCH_SIZE = 32
    DISCOUNT_FACTOR = 0.99
    TARGET_UPDATE_FREQ = 4
    REG_FACTOR = 0.001
    LOG_DIR = '/tmp/dqn'
    random_action_prob = 1
    replay_memory = []
    experiment_filename = './cartpole-experiment-1'
    freqMult = 1
    returns = []
    total_steps = 0
    step_counts = []
    norm_step = []
    norm_return = []
    NUM_EPISODES = 1050
    eta = 0.4

    def __init__(self, env):
        self.env = wrappers.Monitor(gym.make(env), self.experiment_filename, force=True)
        assert len(self.env.observation_space.shape) == 1
        self.input_size = self.env.observation_space.shape[0]
        self.output_size = self.env.action_space.n

    def init_network(self):
        # Inference
        self.x = tf.placeholder(tf.float32, [None, self.input_size])

        with tf.name_scope('hidden1'):
            W1 = tf.Variable(tf.truncated_normal([self.input_size, self.HIDDEN1_SIZE], stddev=0.01), name='W1')
            b1 = tf.Variable(tf.zeros(self.HIDDEN1_SIZE), name='b1')
            h1 = tf.nn.relu(tf.matmul(self.x, W1) + b1)

        with tf.name_scope('hidden2'):
            W2 = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE], stddev=0.01), name='W2')
            b2 = tf.Variable(tf.zeros(self.HIDDEN2_SIZE), name='b2')
            h2 = tf.nn.relu(tf.matmul(h1, W2) + b2)

        with tf.name_scope('output'):
            W3 = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, self.output_size], stddev=0.01), name='W3')
            b3 = tf.Variable(tf.zeros(self.output_size), name='b3')
            self.Q = tf.matmul(h2, W3) + b3
        self.weights = [W1, b1, W2, b2, W3, b3]

        # Defining Loss
        self.targetQ = tf.placeholder(tf.float32, [None])
        self.targetActionMask = tf.placeholder(tf.float32, [None, self.output_size])
        q_values = tf.reduce_sum(tf.multiply(self.Q, self.targetActionMask), reduction_indices=[1])
        temploss = tf.add(tf.subtract(q_values, self.targetQ),
                          self.eta * tf.subtract(q_values, tf.reduce_max(q_values, axis=0)))
        self.loss = tf.reduce_mean(tf.square(temploss))

        # Regularization
        # for w in [W1, W2, W3]:
        #  self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

        # Training
        optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        self.train_op = optimizer.minimize(self.loss, global_step=global_step)

    def train(self, num_episodes=NUM_EPISODES):
        self.session = tf.Session()

        # Summary for TensorBoard
        tf.summary.scalar('loss', self.loss)
        self.summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter(logdir=self.LOG_DIR, graph=self.session.graph)
        self.session.run(tf.global_variables_initializer())

        target_weights = self.session.run(self.weights)

        for episode in range(num_episodes):
            state = self.env.reset()
            steps = 0
            episodic_return = 0

            for step in range(self.MAX_STEPS):
                # Pick the next action and execute it
                action = None
                if random.random() < self.random_action_prob:
                    action = self.env.action_space.sample()
                else:
                    q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
                    action = q_values.argmax()
                self.update_random_action_prob()
                obs, reward, done, _ = self.env.step(action)
                episodic_return += reward

                # Update replay memory
                if done:
                    reward = -20
                self.replay_memory.append((state, action, reward, obs, done))
                if len(self.replay_memory) > self.REPLAY_MEMORY_SIZE:
                    self.replay_memory.pop(0)

                state = obs

                # Sample a random minibatch and fetch max Q at s'
                if len(self.replay_memory) >= self.MINIBATCH_SIZE:
                    minibatch = random.sample(self.replay_memory, self.MINIBATCH_SIZE)
                    next_states = [m[3] for m in minibatch]

                    # ToDo: Optimize to skip terminal states
                    feed_dict = {self.x: next_states}
                    feed_dict.update(zip(self.weights, target_weights))
                    q_values = self.session.run(self.Q, feed_dict=feed_dict)
                    max_q_values = q_values.max(axis=1)

                    # Compute target Q values
                    target_q = np.zeros(self.MINIBATCH_SIZE)
                    target_action_mask = np.zeros((self.MINIBATCH_SIZE, self.output_size), dtype=int)

                    # Update here to change targets
                    for i in range(self.MINIBATCH_SIZE):
                        _, action, reward, _, terminal = minibatch[i]
                        target_q[i] = reward
                        if not terminal:
                            target_q[i] += self.DISCOUNT_FACTOR * max_q_values[i]
                        target_action_mask[i][action] = 1

                    # Gradient descent
                    states = [m[0] for m in minibatch]
                    feed_dict = {
                        self.x: states,
                        self.targetQ: target_q,
                        self.targetActionMask: target_action_mask,
                    }
                    _, summary = self.session.run([self.train_op, self.summary], feed_dict=feed_dict)

                    # Write summary for TensorBoard
                    if self.total_steps % 100 == 0:
                        self.summary_writer.add_summary(summary, self.total_steps)

                self.total_steps += 1
                steps += 1
                if done:
                    episodic_return += reward
                    self.returns.append(episodic_return)
                    break

            self.step_counts.append(steps)
            # mean_steps = np.mean(step_counts[-100:])
            # print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}".format(episode, total_steps, mean_steps))

            # Update target network
            if episode % self.TARGET_UPDATE_FREQ == 0:
                target_weights = self.session.run(self.weights)

        self.norm_step = [self.step_counts]  # [i/( self.freqMult) for i in self.step_counts] # Normalization
        self.norm_return = [self.returns]  # [i/(self.freqMult) for i in self.returns] # Normalization
        print "max steps : ", findMax(self.norm_step)
        print "mean steps : ", np.mean(self.norm_step)
        print "max returns : ", findMax(self.norm_return)
        print "mean returns : ", np.mean(self.norm_return)
        # plotster(norm_step)
        # plotster(norm_return)

    def update_random_action_prob(self):
        self.random_action_prob *= self.RANDOM_ACTION_DECAY
        if self.random_action_prob < self.MIN_RANDOM_ACTION_PROB:
            self.random_action_prob = self.MIN_RANDOM_ACTION_PROB

    def play(self):
        state = self.env.reset()
        done = False
        steps = 0
        total_reward = 0
        while not done and steps < 500:
            # self.env.render()
            q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
            action = q_values.argmax()
            state, rew, done, _ = self.env.step(action)
            steps += 1
            total_reward += rew
        return steps, total_reward


# ENV_LIST = ['CartPole-v1', 'Acrobot-v1', 'MountainCar-v0', 'MountainCarContinuous-v0','Pendulum-v0',
        # 'InvertedPendulum-v1', 'InvertedDoublePendulum-v1', 'Reacher-v1', 'HalfCheetah-v1', 'Swimmer-v1', '
        # Hopper-v1','Walker2d-v1', 'Ant-v1'. 'Humanoid-v1','HumanoidStandup-v1']

def main():
    ENV_NAME = 'Acrobot-v1'
    dqn = DQN(ENV_NAME)
    trial = []
    trial_steps = []
    # dqn.init_network()
    dqn.init_network()

    nb_trials = 3
    for trialNum in xrange(0, nb_trials):
        dqn.train()
        dqn.env.close()
        gym.upload(DQN.experiment_filename, api_key='sk_MD4yPVVeRbCwnaorTY4LuQ')

    temp = np.linspace(0, dqn.NUM_EPISODES - 1, dqn.NUM_EPISODES)
    trial_steps = np.ravel(dqn.norm_step)

    tempo = []
    trial = []

    for i in xrange(0, nb_trials):
        tempo = np.append(tempo, temp)
        trial.extend([i for j in temp])

    trial_returns = np.ravel(dqn.norm_return)
    tempo = (np.ravel(tempo))

    print np.shape(tempo)
    print np.shape(trial)
    print np.shape(trial_steps)
    print np.shape(trial_returns)

    data = pd.DataFrame({'Episode': list(tempo), 'Trial' : list(trial), 'Steps' : list(trial_steps), 'Returns' : list(trial_returns)})
    data.head()

main()


