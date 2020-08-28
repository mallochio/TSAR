import gym
import random
from gym import wrappers
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
plt.style.use('ggplot')

########################################################

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

############################################################

class DQN:
  REPLAY_MEMORY_SIZE     = 1000000
  RANDOM_ACTION_DECAY    = 0.9999
  MIN_RANDOM_ACTION_PROB = 0.1
  HIDDEN1_SIZE           = 64  # 32
  HIDDEN2_SIZE           = 32   # 64
  # HIDDEN3_SIZE           = 512  # 512
  NUM_EPISODES           = 1500
  MAX_STEPS              = 10000000
  LEARNING_RATE          = 0.00025
  MINIBATCH_SIZE         = 32
  DISCOUNT_FACTOR        = 0.99
  TARGET_UPDATE_FREQ     = 4
  REG_FACTOR             = 0.001
  LOG_DIR                = '/tmp/dqn'
  random_action_prob     = 1
  replay_memory          = []
  experiment_filename    = './cartpole-experiment-1'
  freqMult = 1

  def __init__(self, env):
      self.env = wrappers.Monitor(gym.make(env), self.experiment_filename, force = True)
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
    self.loss = tf.reduce_mean(tf.square(tf.subtract(q_values, self.targetQ)))

    # Regularization
    #for w in [W1, W2, W3]:
    #  self.loss += self.REG_FACTOR * tf.reduce_sum(tf.square(w))

    # Training
    optimizer = tf.train.AdamOptimizer(self.LEARNING_RATE)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    self.train_op = optimizer.minimize(self.loss, global_step=global_step)


  def train(self, num_episodes=NUM_EPISODES):
    self.session = tf.Session()

    # Summary for TensorBoard
    tf.summary.scalar('loss', self.loss)
    self.summary        = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(logdir = self.LOG_DIR, graph = self.session.graph)
    self.session.run(tf.global_variables_initializer())
    total_steps         = 0
    step_counts         = []
    target_weights      = self.session.run(self.weights)
    returns = []

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
          if total_steps % 100 == 0:
            self.summary_writer.add_summary(summary, total_steps)

        total_steps += 1
        steps += 1
        if done:
          episodic_return += reward
          returns.append(episodic_return)
          break

      step_counts.append(steps)
      #mean_steps = np.mean(step_counts[-100:])
      #print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}".format(episode, total_steps, mean_steps))

      # Update target network
      if episode % self.TARGET_UPDATE_FREQ == 0:
        target_weights = self.session.run(self.weights)

    norm_step = [i/(500.0 * self.freqMult) for i in step_counts] # Normalization
    norm_return = [i/(500.0 * self.freqMult) for i in returns] # Normalization
    print "max steps : ", findMax(norm_step)
    print "mean steps : ", np.mean(norm_step)
    print "max returns : ", findMax(norm_return)
    print "mean returns : ", np.mean(norm_return)
    plotster(norm_step)
    plotster(norm_return)


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
      #self.env.render()
      q_values = self.session.run(self.Q, feed_dict={self.x: [state]})
      action = q_values.argmax()
      state, rew , done, _ = self.env.step(action)
      steps += 1
      total_reward += rew
    return steps, total_reward


if __name__ == '__main__':
  dqn = DQN('CartPole-v1')
  dqn.init_network()
  dqn.train()
  dqn.env.close()
  gym.upload( DQN.experiment_filename, api_key = 'sk_MD4yPVVeRbCwnaorTY4LuQ')
'''
  res = []
  tot=[]
  for i in range(100):
    #print i, " Im here"
    steps, rew = dqn.play()
    print "Test steps = ", steps, " Total Reward = ", rew
    res.append(steps)
    tot.append(rew)
  print "Mean steps = ",  sum(res)/len(res), " Mean reward = ", sum(tot)/len(tot)
  norm_tot = [i/(500.0 * DQN.freqMult) for i in tot]
  plotster(norm_tot)
'''
