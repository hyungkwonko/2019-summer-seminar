# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import tensorflow as tf
import pickle # to save trained parameter
import random
from collections import deque
import gym
from gym import wrappers # to save .mp4 file
from matplotlib import pyplot as plt
import sys

# HYPER PARAMETER SETTING
REPLAY_MEMORY_SIZE = 5000 # 500000
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 20 # 10000
DISCOUNT_FACTOR = 0.95 # 0.99
ACTION_REPEAT = 4
#UPDATE_FREQUENCY = 4 # ?
LEARNING_RATE = 0.01 # 0.00025
EXPLORATION = 1




# Used Deterministic-v4 (action is selected for every 4 frames)
# https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L352
env = gym.make("PongDeterministic-v4")
#env = gym.make("MontezumaRevengeDeterministic-v4")

# record the game as as an mp4 file
# how to use: https://gym.openai.com/evaluations/eval_lqShqslRtaJqR9yWWZIJg/
env = wrappers.Monitor(env, './pngpnge',  force=True, video_callable=lambda episode_id: episode_id%1==0) 

# dimension check
#input_dim = env.observation_space.shape # (210, 160, 3)
#input_h = p_obs.shape[0]
#input_w = p_obs.shape[1]
input_h = 80
input_w = 80
input_size = input_h * input_w
output_size = env.action_space.n # 6

# Possible action set
env.unwrapped.get_action_meanings() # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']


class DQN:
    
    def __init__(self, session, input_h, input_w, output_size, name="main"):
        self.session = session
        self.input_h = input_h # 80
        self.input_w = input_w # 80
        self.input_size = 80 * 80 * 4 # 6400
        self.output_size = output_size
        self.net_name = name
        self._build_network()
    
    def _build_network(self, h_size=10, l_rate=1e-1):
        with tf.variable_scope(self.net_name):

            self._X = tf.placeholder(tf.float32, [None,80*80*4])

            # input 을 이미지로 인식하기 위해 reshape을 해준다. 80*80의 이미지이며 4개의 채널, 개수는 n개이므로 -1
            self._X_img = tf.reshape(self._X, [-1,80,80,4])

            # layer 1
            # SIZE: 80x80, CHANNEL: 4, FILTER: 16 (8x8)
            W1 = tf.Variable(tf.random_normal([8,8,4,16], stddev=0.1))

            # STRIDE: 4x4, PADDING: same
            L1 = tf.nn.conv2d(self._X_img, W1, strides=[1,4,4,1], padding='SAME')
            L1 = tf.nn.relu(L1)
            
            # layer 2
            # SIZE: 20x20, CHANNEL: 16, FILTER: 32 (4x4)
            W2 = tf.Variable(tf.random_normal([4,4,16,32], stddev = 0.1))
            
            # STRIDE: 2x2, PADDING: same
            L2 = tf.nn.conv2d(L1, W2, strides=[1,2,2,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            
            # SIZE: 10x10, CHANNEL: 32 --> 10x10x32= 3200
            L2 = tf.reshape(L2, [-1,3200])
            
            # fully-connected layer
            W3 = tf.get_variable("W3", shape=[3200, 256], initializer=tf.contrib.layers.xavier_initializer())            
            L3 = tf.nn.tanh(tf.matmul(L2, W3))
            
            # output layer 
            W4 = tf.get_variable("W4", shape=[256, 6], initializer=tf.contrib.layers.xavier_initializer())        
            self._Qpred = tf.matmul(L3, W4)
        
        # Policy
        self._Y = tf.placeholder(shape=[None, 6], dtype=tf.float32)  # 6가지의 output cases

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        
    def predict(self, state):
        x = state.reshape([-1, 80*80*4])
#        x = np.reshape(state, [1, self.input_size])
        return self.session.run(self._Qpred, feed_dict={self._X: x})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})  


def replay_train(mainDQN, targetDQN, minibatch):
    x_stack = np.empty(0).reshape(-1, 6400*4) # array([], shape=(0, input_size), dtype=float64)
    y_stack = np.empty(0).reshape(-1, 6) # array([], shape=(0, output_size), dtype=float64)
    
    for state, action, reward, next_state, done in minibatch:
        Q = mainDQN.predict(state)
        
        if(done):
            Q[0, action] = reward
        else:
            Q[0, action] = reward + DISCOUNT_FACTOR * np.max(targetDQN.predict(next_state))

        x_stack = np.vstack([x_stack, state]) # x_stack = (-1, 64000), state = (320*80)
        y_stack = np.vstack([y_stack, Q])
    
    return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))

    return op_holder


def bot_play(mainDQN):
    s = env.reset()
    reward_sum = 0
    while(True):
        env.render()
        a = np.argmax(mainDQN.predict(s))
        s, reward, done, _ = env.step(a)
        reward_sum += reward
        if(done):
            print("Total score: {}".format(reward_sum))
            break


def preprocess_test(env):
    # reset environment
    act = env.action_space.sample()
    
    for _ in range(50):
        img, _, _, _ = env.step(act)
    
    # original image
    plt.imshow(img)
    plt.show()

    # remove 3rd dimension
    img = img[35:195] # crop
    img = img[::2, ::2] # ex) 160x160 -> 80x80
    img = img[:,:,0] # only leave 'R' from R-G-B    

    # remove background
    np.unique(img)
    img[img == np.bincount(list(img.reshape(-1))).argmax()] = 0 # erase background    

    # set other colors simple
    img[img != 0] = 1 # set others = 1
    
    # print out image
    plt.imshow(img, cmap='gray')
    plt.show()
    print("Please check if it's working correct")


def preprocess(img):
    img = img[35:195] # crop
    img = img[::2, ::2] # ex) 160x160 -> 80x80
    img = img[:,:,0] # only leave 'R' from R-G-B    
    img[img == np.bincount(list(img.reshape(-1))).argmax()] = 0 # erase background    
    img[img != 0] = 1 # set others = 1
    return img.reshape(80,80,1)


if __name__ == "__main__":
    max_episodes = 5000
    preproc = False
    
    # test preprocessing
    if(preproc):
        env.reset()
        preprocess_test(env)
        sys.exit(0)
    
    # line 1 (initialize replay memory D to capacity N)
    replay_buffer = deque()
    
    with tf.Session() as sess:
#        sess = tf.InteractiveSession();
#        sess.close()
        mainDQN = DQN(sess, input_h, input_w, output_size, name="main")
        targetDQN = DQN(sess, input_h, input_w, output_size, name="target")
        tf.global_variables_initializer().run()
        
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        
        sess.run(copy_ops)
    
        for episode in range(max_episodes):
            # Exploration variable: [Initial = 1, Final = 0.1]
            EXPLORATION = 0.9 / ((episode / 10) + 1) + 0.1
            done = False
            reward_total = 0
            
            # line 5-1 (Initialize sequence s_q = {x_1})
            state = env.reset()
    
            # line 5-2 (preprocessed sequenced phi_1 = phi(s_1))
            state = state2 = state3 = state4 = preprocess(state) # preprocess state (210, 160, 3) -> (80, 80)
    
            # line 6 (for t = 1,T do)
            while not done: 
                
                # line 7 (With probability e select a random action a_t)
                if(np.random.rand(1) < EXPLORATION):
                    action = env.action_space.sample()
                else: # line 8 (otherwise select a_t = max ...)
                    tmp = np.concatenate((state, state2, state3, state4), axis=2)
                    action = np.argmax(mainDQN.predict(tmp))
#                    action = np.argmax(mainDQN.predict(state))

                # line 9 (execute action a_t in emulator and observe reward r_t and image x_t+1)                    

                # get next states, rewards, dones
                next_state, reward, done, _ = env.step(action)
                if(done):
                    next_state2 = next_state3 = next_state4 = next_state
                    reward2 = reward3 = reward4 = 0
                    done2 = done3 = done4 = done
                else:
                    next_state2, reward2, done2, _ = env.step(action)
                    if(done2):
                        next_state3 = next_state4 = next_state2
                        reward3 = reward4 = 0
                        done3 = done4 = done2
                    else:
                        next_state3, reward3, done3, _ = env.step(action)
                        if(done3):
                            next_state4 = next_state3
                            reward4 = 0
                            done4 = done3
                        else:
                            next_state4, reward4, done4, _ = env.step(action)


                # preprocess next_state also
                next_state = preprocess(next_state)
                next_state2 = preprocess(next_state2)
                next_state3 = preprocess(next_state3)
                next_state4 = preprocess(next_state4)
                
                # 여기서 replay buffer 에 집어넣은 state와 next_state 값을 4배로 불려서 넣어줘야 한다.            
#                replay_buffer.append((state, action, reward, next_state, done))
                s_t = np.concatenate((state, state2, state3, state4), axis=2).reshape(-1,25600)
                s2_t = np.concatenate((next_state, next_state2, next_state3, next_state4), axis=2).reshape(-1,25600)

                reward_t = reward + reward2 + reward3 + reward4
                done = done or done2 or done3 or done4

                replay_buffer.append((s_t, action, reward_t, s2_t, done))
    
                if(len(replay_buffer) > REPLAY_MEMORY_SIZE):
                    replay_buffer.popleft()

                # update state                    
                state = next_state
                state2 = next_state2
                state3 = next_state3
                state4 = next_state4

                reward_total = reward_total + reward_t
                                
            print("Episode: {}, reward_total: {}".format(episode, reward_total))
            
            # stop looping if total reward is bigger than 10
            if(reward_total > 10):
                break
            
            if(episode % 100 == 1): # should be 10000
                for _ in range(50):
                    minibatch = random.sample(replay_buffer, 32)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                print("Loss: ", loss)
                
                # copy network from mainDQN to targetDQN
                sess.run(copy_ops)
    
#            bot_play(mainDQN)    
    