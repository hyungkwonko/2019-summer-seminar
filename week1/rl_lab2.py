# -*- coding: utf-8 -*-

"""
@source: https://www.youtube.com/watch?v=xvDAURQVDhk&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=3
@written by: Sung KIM
@modified by: Hyung-Kwon Ko
@created on: Jul 11 17:37:49 2019
@last modified date: 07/11/19
"""

import gym
from gym.envs.registration import register
import colorama as cr

# set MACRO as global var
LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
    'w' : UP,
    's' : DOWN,
    'd' : RIGHT,
    'a' : LEFT
}

# setting module
class Lab2:
    def __init__(self, id, ep):
        cr.init(autoreset=True)
        gym.envs.registration.register(id=id, entry_point=ep, kwargs={'map_name':'4x4', 'is_slippery':False})
        self.env = gym.make(id)
        self.env.render()

# run program
if __name__ == "__main__":
    env = Lab2('FrozenLake-v3', 'gym.envs.toy_text:FrozenLakeEnv').env

    while True:
        # get char with input()
        key = input()
        if key not in arrow_keys.keys():
            # quit program if == 'q'
            if key == 'q':
                print("Quit program")
                break
            # continue if undefined key
            else:
                print("please check keyset")
                continue
    
        action = arrow_keys[key]
        state, reward, done, info = env.step(action)
        env.render()
        print("State: ", state, ", Action: ", action, ", Reward: ", reward, ", Info: ", info)
        # quit if done == 1
        if done:
            print("Finished with reward", reward)
            break