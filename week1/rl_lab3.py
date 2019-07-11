# -*- coding: utf-8 -*-
"""
@source: https://www.youtube.com/watch?v=yOBKtGU6CG0&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=5
@written by: Sung KIM
@modified by: Hyung-Kwon Ko
@created on: Jul 11 18:33:00 2019
@last modified date: 07/11/19

"""

    
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):    # https://gist.github.com/stober/1943451
    # Argmax that chooses randomly among eligible maximum idices.
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

class Lab3:
    def __init__(self, id, ep):
        gym.envs.registration.register(id=id, entry_point=ep, kwargs={'map_name':'4x4', 'is_slippery':False})
        self.env = gym.make(id)
        self.env.render()

# run program
if __name__ == "__main__":
    env = Lab3('FrozenLake-v3', 'gym.envs.toy_text:FrozenLakeEnv').env
    
    # Initialize table with all zeros
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    # Set learning parameters
    num_episodes = 2000
    # create lists to contain total rewards and steps per episode
    rList = []
    for i in range(num_episodes):
        # Reset environment and get first new observation
        state = env.reset()
        rAll = 0
        done = False
    
        # The Q-Table learning algorithm
        while not done:
            action = rargmax(Q[state, :])
    
            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)
    
            # Update Q-Table with new knowledge using learning rate
            Q[state, action] = reward + np.max(Q[new_state, :])

            rAll += reward
            state = new_state
        rList.append(rAll)
    
    print("Success rate: " + str(sum(rList) / num_episodes))
    print("Final Q-Table Values")
    print("LEFT DOWN RIGHT UP")
    print(Q)
    
    plt.bar(range(len(rList)), rList, color="blue")
    #plt.bar(range(len(rList)), rList, color='b', alpha=0.4)
    plt.show()