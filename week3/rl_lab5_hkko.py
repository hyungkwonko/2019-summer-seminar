# -*- coding: utf-8 -*-

"""
@source: https://www.youtube.com/watch?v=VYOq-He90bE&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=7
@written by: Sung KIM
@modified by: Hyung-Kwon Ko
@created on: Jul 17 19:03 2019
@last modified date: 2019-07-19
"""

# import required packages
import numpy as np
import gym
import matplotlib.pyplot as plt
from gym.envs.registration import register
    
# parameter setting
gamma = 0.99 # discounted reward (factor)
num_episodes = 2000 # number of iteration
learning_rate = 0.85 # learning rate

# setting module
class Lab5:
    def __init__(self, id):
        self.env = gym.make(id)
        
    #  Use E & E as default, but can change by setting noise = 1
    def algorithm(self, iter = 2000, noise = 0, gamma = 0.99, lr = 0.85):
        '''
        @param iter: number of iteration
        @param noise: E&E or Noise
        @param gamma: discounted factor
        @return Q: Q-Table
        @return rList: set of reward as list
        '''
                
        # Create lists to contain total rewards and steps per episode
        rList = []

        #Initialize table with all zeros
        Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        for i in range(iter):
            # Reset environment and get first new observation
            state = self.env.reset()
            rAll = 0
            done = False
            
            # 'e' is for E & E algorithm
            e = 1. / ((i / 100) + 1)
        
            # The Q-Table learning algorithm
            while not done:
                if(noise):
                    # Add noise - Choose an action by greedily (with noise) picking from Q table
                    action = np.argmax(Q[state, :] + np.random.randn(1, self.env.action_space.n) / (i+1))
                else:
                    # Choose an action by e value
                    if(np.random.rand(1) < e):
                        action = self.env.action_space.sample()
                    else:
                        action = np.argmax(Q[state, :])
                
                # Get new state and reward from environment
                new_state, reward, done, _ = self.env.step(action)
                
                # Update Q table
                Q[state, action] = (1-lr) * Q[state, action] + lr*(reward+ gamma*np.max(Q[new_state, :]))

                rAll += reward
                state = new_state
            rList.append(rAll)
        return Q, rList

# run program
if __name__ == "__main__":

    # Set env
    l5 = Lab5('FrozenLake-v0')

    # User chooses
    noise = input("E/E(0) or Noise(1)? ")
    
    # Run algorithm
    Q, rList = l5.algorithm(iter = num_episodes, noise = int(noise), gamma = gamma, lr=learning_rate)

    # Print out Success rate and Q-Table
    print("Success rate: ", str(sum(rList)/num_episodes))
    print("Final Q-Table Values")
    print(Q)

    # Outcome of the discounted factor is shown well
    plt.bar(range(len(rList)), rList, color="blue")
    plt.show()
