### MDP Value Iteration and Policy Iteration
### Acknowledgement: start-up codes were adapted with permission from Prof. Emma Brunskill of Stanford University

import numpy as np
import gym
import time
from lake_envs import *
from _asyncio import Future
from numpy.distutils.cpuinfo import key_value_from_command

np.set_printoptions(precision=3)

"""
For policy_evaluation, policy_improvement, policy_iteration and value_iteration,
the parameters P, nS, nA, gamma are defined as follows:

	P: nested dictionary
		From gym.core.Environment
		For each pair of states in [1, nS] and actions in [1, nA], P[state][action] is a
		tuple of the form (probability, nextstate, reward, terminal) where
			- probability: float
				the probability of transitioning from "state" to "nextstate" with "action"
			- nextstate: int
				denotes the state we transition to (in range [0, nS - 1])
			- reward: int
				either 0 or 1, the reward for transitioning from "state" to
				"nextstate" with "action"
			- terminal: bool
			  True when "nextstate" is a terminal state (hole or goal), False otherwise
	nS: int
		number of states in the environment
	nA: int
		number of actions in the environment
	gamma: float
		Discount factor. Number in range [0, 1)
"""

def policy_evaluation(P, nS, nA, policy, gamma=0.9, tol=1e-3):
	"""Evaluate the value function from a given policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	policy: np.array[nS]
		The policy to evaluate. Maps states to actions.
	tol: float
		Terminate policy evaluation when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns
	-------
	value_function: np.ndarray[nS]
		The value function of the given policy, where value_function[s] is
		the value of state s
	"""

	value_function = np.zeros(nS)

	############################
	# YOUR IMPLEMENTATION HERE #
	MAX_ITERATION = 100
	
	for iterator in range(MAX_ITERATION): # loop for maximizing value_function, called value iteration
		value_new = np.zeros(nS) # initialize value_new array as zeros
		delta = tol # not set delta to zero to avoid (delta < tol) true at first loop
		for s in range(nS): # in all States, here are 0 - 15, totally 16 states
			vs = 0 # saves the value of V(s) for each action
			for a in [policy[s]]: # in all actions in the sequence
				# The structure of P is P[s][a] = [(prob, next_state, reward, terminal), ...]
				# prob is probability, double type
				# next_state is the next state this action will lead to, int type
				# reward is the current reward in this state, double type
				# terminal is a boolean flag to show if the current state is a terminal state, bool type
				for prob, next_state, reward, terminal in P[s][a]: # loop to add all future values of all next_state can get to together
					if terminal:
						vs += prob * reward # if the current state is a terminal state, the future reward is 0
					else:
						vs += prob * (reward + gamma * value_function[next_state]) # else, future reward is gamma * value_function[next_state]
			value_new[s] = vs # save the total value of all future next_state (possible states can get to) to value_new[s] because vs is temporary and will be initialized to zero
		delta = max(delta, abs(value_new[s] - value_function[s])) # get the bigger one between delta and |value_new[s] - value_function[s]| to check if value_ is not changing (stable)
		if delta < tol: break # when value_ is not changing, break from the outer loop, not iterating anymore
		value_function = value_new # save value_new to value_function, which is value_old
	############################
	return value_function


def policy_improvement(P, nS, nA, value_from_policy, policy, gamma=0.9):
	"""Given the value function from policy improve the policy.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	value_from_policy: np.ndarray
		The value calculated from the policy
	policy: np.array
		The previous policy.

	Returns
	-------
	new_policy: np.ndarray[nS]
		An array of integers. Each integer is the optimal action to take
		in that state according to the environment dynamics and the
		given value function.
	"""

	new_policy = np.zeros(nS, dtype='int')

	############################
	# YOUR IMPLEMENTATION HERE #
	for s in range(nS):
		maxvsa = 0 # saves the max value of V(s)
		for a in range(nA):
			vsa = 0 # saves the temporary value of V(s) for all possible future states if take action a now
			for prob, next_state, reward, terminal in P[s][a]:
				if terminal:
					vsa += prob * reward
				else:
					vsa += prob * (reward + gamma * value_from_policy[next_state])
				if vsa > maxvsa: # if get a new max
					maxvsa = vsa # update the maxvsa
					new_policy[s] = a # update this step's action to a
	############################
	return new_policy


def policy_iteration(P, nS, nA, gamma=0.9, tol=10e-3):
	"""Runs policy iteration.

	You should call the policy_evaluation() and policy_improvement() methods to
	implement this method.

	Parameters
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		tol parameter used in policy_evaluation()
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)

	############################
	# YOUR IMPLEMENTATION HERE #
	MAX_ITERATION = 100
	
	old_policy = policy # declare old_policy to compare if the policy has changed, target is no change anymore
	for iterator in range(MAX_ITERATION):
		value_function = policy_evaluation(P, nS, nA, policy, gamma, tol) # value iteration with the always updating policy
		policy = policy_improvement(P, nS, nA, value_function, policy, gamma) # update the policy with the new value function
		if np.array_equal(policy, old_policy): break # if policy is stable, break the policy iteration, target policy got
		old_policy = policy # if the policy is still not stable, need to update the old one before loop again
	############################
	return value_function, policy

def value_iteration(P, nS, nA, gamma=0.9, tol=1e-3):
	"""
	Learn value function and policy by using value iteration method for a given
	gamma and environment.

	Parameters:
	----------
	P, nS, nA, gamma:
		defined at beginning of file
	tol: float
		Terminate value iteration when
			max |value_function(s) - prev_value_function(s)| < tol
	Returns:
	----------
	value_function: np.ndarray[nS]
	policy: np.ndarray[nS]
	"""

	value_function = np.zeros(nS)
	policy = np.zeros(nS, dtype=int)
	############################
	# YOUR IMPLEMENTATION HERE #
	MAX_ITERATION = 1000
	
	value_new = np.zeros(nS) # initialize value_new array as zeros
	for iterator in range(MAX_ITERATION): # loop for maximizing value_function, called value iteration
		delta = tol # not set delta to zero to avoid (delta < tol) true at first loop
		for s in range(nS):
			maxvsa = 0 # saves the max value of V(s)
			for a in range(nA):
				vsa = 0 # saves the temporary value of V(s) for all possible future states if take action a now
				for prob, next_state, reward, terminal in P[s][a]:
					if terminal:
						vsa += prob * reward
					else:
						vsa += prob * (reward + gamma * value_function[next_state])
					if vsa > maxvsa:
						maxvsa = vsa
						policy[s] = a # update this step's action to a because a new max V(s) of all futures is get
			delta = max(delta, abs(value_function[s] - maxvsa)) # get the bigger one between delta and |value_new[s] - value_function[s]| to check if value_ is not changing (stable)
			value_new[s] = maxvsa # save the new max value of state s to value_new array
		if delta >= tol:
			value_function = value_new # if there is still difference between old and new values, update the old value, prepare to loop again
		else:
			break # break from value iteration, optimized value and policy got
	############################
	return value_function, policy

def render_single(env, policy, max_steps=100):
  """
    This function does not need to be modified
    Renders policy once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play on. Must have nS, nA, and P as
      attributes.
    Policy: np.array of shape [env.nS]
      The action to take at a given state
  """

  episode_reward = 0
  ob = env.reset()
  for t in range(max_steps):
    env.render()
    time.sleep(0.25)
    a = policy[ob]
    ob, rew, done, _ = env.step(a)
    episode_reward += rew
    if done:
      break
  env.render();
  if not done:
    print("The agent didn't reach a terminal state in {} steps.".format(max_steps))
  else:
  	print("Episode reward: %f" % episode_reward)


# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
# You may change the parameters in the functions below
if __name__ == "__main__":

	# comment/uncomment these lines to switch between deterministic/stochastic environments
	#env = gym.make("Deterministic-4x4-FrozenLake-v0")
	env = gym.make("Stochastic-4x4-FrozenLake-v0")

	print("\n" + "-"*25 + "\nBeginning Policy Iteration\n" + "-"*25)

	V_pi, p_pi = policy_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_pi, 100)

	print("\n" + "-"*25 + "\nBeginning Value Iteration\n" + "-"*25)

	V_vi, p_vi = value_iteration(env.P, env.nS, env.nA, gamma=0.9, tol=1e-3)
	render_single(env, p_vi, 100)
	
	


