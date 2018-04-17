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

	# CS234, Spring 2017
	# Hyun Sik Kim
	# Assignment 3. Coding part
	# hsik, hsik@stanford.edu

	result = np.zeros(10000)
	summation = 0.0;

	# generating episodes
	for k in range(num_episodes):

		# start a new episode
		s = env.reset()
		episode_reward = 0.0
		while True:
			# selecting action
			action = np.argmax([Q[s, a] for a in range(env.nA)])
			# getting the next state and corresponding reward
			nextS, reward, done, _ = env.step(action)

			# updating reward
			episode_reward += reward

			# Rmax implementation
			if nSA[s, action] < m:
				# updating the knowledge
				nSA[s, action] = nSA[s, action] + 1
				R[s, action] = R[s, action] + reward
				nSASP[s, action, nextS] = nSASP[s, action, nextS] + 1

				if nSA[s, action] == m:
					for i in range(max_step):
						for State in range(env.nS):
							for Action in range(env.nA):
								if nSA[State, Action] >= m:
									# update the estimation of action-value functions
									Q[State, Action] = R[State, Action] / float(nSA[State, Action])
									Q[State, Action] = Q[State, Action] + gamma * np.sum([nSASP[State, Action, Next_S] / float(nSA[State, Action]) * np.max([Q[Next_S, a_p] for a_p in range(env.nA)]) for Next_S in range(env.nS)])
									print(Q[State, Action])
			s = nextS

			if done:
				break

		# for the purpose of plot
		if k < 10000:
			summation = summation + episode_reward
			result[k] = summation / (k + 1)

	plt.plot(result)
	plt.ylabel('Average Score')
	plt.xlabel('# episodes')
	plt.title('Rmax, problem 1.(b)')
	plt.show()

	########################################################
	#                    END YOUR CODE                     #
	########################################################
	return Q


def main():
	env = FrozenLakeEnv(is_slippery=False)
	print env.__doc__
	Q = rmax(env, gamma = 0.99, m=10, R_max = 1, epsilon = 0.1, num_episodes = 1000)
	render_single_Q(env, Q)


if __name__ == '__main__':
	print "haha"
	main()