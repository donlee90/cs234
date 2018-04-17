### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.99, lr=0.85, e=0.85, decay_rate=0.999):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################
  Q = np.zeros((env.nS, env.nA))

  # Generate episodes
  average_scores = []
  accum = 0.0
  for i in xrange(num_episodes):
    S = env.reset()
    done = False
    episode_reward = 0

    while not done:
      # Epsilon-greedy choice of action
      if np.random.rand() < e:
        A = env.action_space.sample()
      else:
        A = np.argmax([Q[S,a] for a in xrange(env.nA)])

      # Make an action
      nextS, R, done, _ = env.step(A)
      episode_reward += R

      # Update Q-value
      Q[S,A] = (1 - lr) * Q[S,A] + lr * (R + gamma * max(Q[nextS, a] for a in xrange(env.nA)))
      S = nextS

    accum += episode_reward
    average_scores.append(accum/(i+1))

    lr = lr * decay_rate
    e = e * decay_rate

  plt.plot(average_scores[:1000])
  plt.xlabel('Episode number')
  plt.ylabel('Average score')
  plt.title('Q-learning')
  plt.show()

  return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.99, lr=0.5, e=0.95, decay_rate=0.999):
  """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state-action values
  """

  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  Q = np.zeros((env.nS, env.nA))

  # Generate episodes
  for i in xrange(num_episodes):
    S = env.reset()
    done = False
    while not done:
      # Epsilon-greedy choice of action
      if np.random.rand() < e:
        A = env.action_space.sample()
      else:
        A = np.argmax([Q[S,a] for a in xrange(env.nA)])

      # Make an action
      nextS, R, done, _ = env.step(A)

      # Get A' following epsilon-greedy strategy
      if np.random.rand() < e:
        nextA = env.action_space.sample()
      else:
        nextA = np.argmax([Q[nextS,a] for a in xrange(env.nA)])
      
      # Update Q-value
      Q[S,A] = (1 - lr) * Q[S,A] + lr * (R + gamma * Q[nextS, nextA])
      S = nextS

    lr = lr * decay_rate
    e = e * decay_rate

  return Q

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward


def run_single_Q(env, Q):
  """Runs Q function once on environment.

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.

    Returns:
      episode_reward
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward
  return episode_reward

def evaluate_performance(env, Q):
  N = 100
  score = 0.0
  for i in range(N):
    score += run_single_Q(env, Q)
  print score/N


# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q = learn_Q_QLearning(env, gamma=0.99, e=0.85, lr=0.85, decay_rate=0.999)
  #Q = learn_Q_SARSA(env, gamma=0.99, e=0.95, lr=0.5, decay_rate=0.999)
  print Q

  #render_single_Q(env, Q)
  evaluate_performance(env, Q)

if __name__ == '__main__':
    main()
