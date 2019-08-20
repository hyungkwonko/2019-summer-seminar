# -*- coding: utf-8 -*-
"""
@author: Hyung-Kwon Ko
@created: Mon Aug 19 21:58:27 2019
@last modified: Mon Aug 19 21:58:27 2019
"""

# Retrived directly from [https://github.com/openai/baselines/blob/master/baselines/deepq/experiments/enjoy_pong.py] just for checking

import gym
from baselines import deepq
from gym import wrappers # to save .mp4 file


def main():
    env = gym.make("PongNoFrameskip-v4")
    
    env = wrappers.Monitor(env, './pngpnge',  force=True, video_callable=lambda episode_id: episode_id%10==0) 

    env = deepq.wrap_atari_dqn(env)
    model = deepq.learn(
        env,
        "conv_only",
        convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
        hiddens=[256],
        dueling=True,
        total_timesteps=0
    )

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            obs, rew, done, _ = env.step(model(obs[None])[0])
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()