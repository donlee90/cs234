import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt

def learn_Q_QLearning(env, num_episodes=10000, gamma = 0.99, lr = 0.1, e = 0.2, max_step=6):
    """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy(no decay)
    Feel free to reuse your assignment1's code
    Parameters
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as attributes.
    num_episodes: int 
        Number of episodes of training.
    gamma: float
        Discount factor. Number in range [0, 1)
    learning_rate: float
        Learning rate. Number in range [0, 1)
    e: float
        Epsilon value used in the epsilon-greedy method. 
    max_step: Int
        max number of steps in each episode

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    Q = np.zeros((env.nS, env.nA))
    ########################################################
    #                     YOUR CODE HERE                   #
    ########################################################

    # Generate episodes
    average_scores = []
    accum = 0.0
    for i in xrange(num_episodes):
        S = env.reset()
        done = False
        episode_reward = 0
        n_steps = 0

        while not done:

            if n_steps >= max_step:
                break

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
            n_steps += 1

        accum += episode_reward
        average_scores.append(accum/(i+1))


    plt.plot(average_scores[:10000], label="epsilon=%f"%(e))


    ########################################################
    #                     END YOUR CODE                    #
    ########################################################
    return Q



def main():
    env = FrozenLakeEnv(is_slippery=False)
    Q = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.1, e = 0.9)
    Q = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.1, e = 0.6)
    Q = learn_Q_QLearning(env, num_episodes = 10000, gamma = 0.99, lr = 0.1, e = 0.3)

    plt.xlabel('Episode number')
    plt.ylabel('Average score')
    plt.title('Q-learning')
    plt.legend()
    plt.show()

    render_single_Q(env, Q)


if __name__ == '__main__':
    main()
