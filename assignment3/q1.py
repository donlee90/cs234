import math
import gym
from frozen_lake import *
import numpy as np
import time
from utils import *
import matplotlib.pyplot as plt


def rmax(env, gamma, m, R_max, epsilon, num_episodes, max_step = 6):
    """Learn state-action values using the Rmax algorithm

    Args:
    ----------
    env: gym.core.Environment
        Environment to compute Q function for. Must have nS, nA, and P as
        attributes.
    m: int
        Threshold of visitance
    R_max: float 
        The estimated max reward that could be obtained in the game
    epsilon: 
        accuracy paramter
    num_episodes: int 
        Number of episodes of training.
    max_step: Int
        max number of steps in each episode

    Returns
    -------
    np.array
      An array of shape [env.nS x env.nA] representing state-action values
    """

    Q = np.ones((env.nS, env.nA)) * R_max / (1 - gamma)
    R = np.zeros((env.nS, env.nA))
    nSA = np.zeros((env.nS, env.nA))
    nSASP = np.zeros((env.nS, env.nA, env.nS))
    ########################################################
    #                   YOUR CODE HERE                     #
    ########################################################

    # Generate episodes
    average_scores = []
    accum = 0.0
    term = int(np.log(1 / (epsilon * (1 - gamma))) / (1 - gamma))
    for i in xrange(num_episodes):
        S = env.reset()
        done = False
        episode_reward = 0.0
        n_steps = 0

        while not done:

            if n_steps >= max_step:
                break

            A = np.argmax([Q[S,a] for a in range(env.nA)])

            # Make an action
            nextS, reward, done, _ = env.step(A)
            episode_reward += reward

            # R-Max
            if nSA[S, A] < m:
                nSA[S, A] += 1
                R[S, A] += reward
                nSASP[S, A, nextS] += 1

                if nSA[S, A] == m:
                    for j in range(term):
                        for S_bar in range(env.nS):
                            for A_bar in range(env.nA):
                                if nSA[S_bar, A_bar] >= m:
                                    N = float(nSA[S_bar, A_bar])
                                    T_hat = nSASP[S_bar, A_bar, :] / N
                                    R_hat = R[S_bar, A_bar] / N
                                    Q[S_bar, A_bar] = R_hat
                                    Q[S_bar, A_bar] += gamma * np.sum(T_hat * np.max(Q, axis=1))


            # Update Q-value
            S = nextS
            n_steps += 1

        accum += episode_reward
        average_scores.append(accum/(i+1))

    plt.plot(average_scores[:10000], label="m=%d"%(m))

    ########################################################
    #                    END YOUR CODE                     #
    ########################################################
    return Q


def main():
    env = FrozenLakeEnv(is_slippery=False)
    print env.__doc__
    Q = rmax(env, gamma = 0.99, m=1, R_max = 1, epsilon = 0.1, num_episodes = 10000)
    Q = rmax(env, gamma = 0.99, m=10, R_max = 1, epsilon = 0.1, num_episodes = 10000)
    Q = rmax(env, gamma = 0.99, m=20, R_max = 1, epsilon = 0.1, num_episodes = 10000)
    Q = rmax(env, gamma = 0.99, m=50, R_max = 1, epsilon = 0.1, num_episodes = 10000)

    plt.xlabel('Episode number')
    plt.ylabel('Average score')
    plt.title('Rmax')
    plt.legend()
    plt.show()


    render_single_Q(env, Q)


if __name__ == '__main__':
    print "haha"
    main()
