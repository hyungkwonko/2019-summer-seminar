# -*- coding: utf-8 -*-
"""
@author: Hyung-Kwon Ko
@created: Wed Jul 31 15:12:16 2019
@last modified: Wed Jul 31 15:12:16 2019
"""

import numpy as np
import tensorflow as tf
import random
from collections import deque
import gym
from gym import wrappers # to save .mp4 file
from matplotlib import pyplot as plt
import sys
import argparse


# HYPER PARAMETER SETTING
MINIBATCH_SIZE = 32
REPLAY_MEMORY_SIZE = 500000
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 10000
TRAIN_START = 50000
DISCOUNT_FACTOR = 0.99 # 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4 # ???
LEARNING_RATE = 0.00025
EXPLORATION = 1 # initial value
FINAL_EXPLORATION_FRAME = 1000000 # 1000000 Should be run after this number, but set it as our final number of steps to run
TOTAL_EPISODE = 500*50
MODEL_SAVE_LOCATION = "c:/users/hkko/desktop/model/model.ckpt"
VIDEO_SAVE_LOCATION = "c:/users/hkko/desktop/pong"
EPSILON = 0.01
MOMENTUM = 0.95

# Used Deterministic-v4 (action is selected for every 4 frames)
# https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L352
env = gym.make("PongDeterministic-v4")
#env = gym.make("MontezumaRevengeDeterministic-v4")

# record the game as as an mp4 file
# how to use: https://gym.openai.com/evaluations/eval_lqShqslRtaJqR9yWWZIJg/
env = wrappers.Monitor(env, VIDEO_SAVE_LOCATION, force=True, video_callable=lambda episode_id: episode_id%50==0) 

# INPUT DIMENSION
#input_size = env.observation_space.shape # (210, 160, 3)
#output_size = env.action_space.n # 6
# Possible action set
#env.unwrapped.get_action_meanings() # ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
INPUT_H = 80
INPUT_W = 80
INPUT_SIZE = INPUT_H * INPUT_W
OUTPUT_SIZE = 6
CHANNEL = 4


class DQN:
    
    def __init__(self, session, input_h, input_w, chan, output_size, name="main"):
        self.session = session
        self.input_h = input_h # 80
        self.input_w = input_w # 80
        self.chan = chan # 4
        self.input_size = 80 * 80 * 4 # 80 * 80 * 4
        self.output_size = output_size
        self.net_name = name
        self._build_network()
      
    # function code retrived from https://passi0n.tistory.com/88
    def cliped_error(self, error):
        return tf.where(tf.abs(error) < 1.0, 0.5 * tf.square(error), tf.abs(error) - 0.5)
     
    def _build_network(self, l_rate=LEARNING_RATE):
        with tf.compat.v1.variable_scope(self.net_name):

            # input, SIZE: (batch_size, 80, 80, 4)
            self._X = tf.compat.v1.placeholder(tf.float32, [None,80,80,4])

            # layer 1
            # SIZE: 80x80, CHANNEL: 4, FILTER: 32 (8x8)
            W1 = tf.compat.v1.get_variable("W1", shape=[8,8,4,32], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            # STRIDE: 4x4, PADDING: VALID
            L1 = tf.nn.relu(tf.nn.conv2d(self._X, W1, strides=[1,4,4,1], padding='VALID'))
            
            # layer 2
            # SIZE: 19x19, CHANNEL: 32, FILTER: 64 (4x4)
            W2 = tf.compat.v1.get_variable("W2", shape=[4,4,32,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            # STRIDE: 2x2, PADDING: VALID
            L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1,2,2,1], padding='VALID'))
            
            # layer 3
            # SIZE: 6x6, CHANNEL: 64, FILTER: 64 (3x3)
            W3 = tf.compat.v1.get_variable("W3", shape=[3,3,64,64], initializer=tf.contrib.layers.xavier_initializer_conv2d())
            # STRIDE: 2x2, PADDING: VALID
            L3 = tf.nn.relu(tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='VALID'))
            L3 = tf.reshape(L3, [-1, 6*6*64])
            
            # Fully-connected layer: 2304 -> 256
            W4 = tf.compat.v1.get_variable("W4", shape=[6*6*64, 256], initializer=tf.contrib.layers.xavier_initializer())            
            L4 = tf.nn.relu(tf.matmul(L3, W4))
            
            # output layer, 256 -> 6
            W5 = tf.compat.v1.get_variable("W5", shape=[256, 6], initializer=tf.contrib.layers.xavier_initializer())        
            self._Qpred = tf.matmul(L4, W5)
        
        # Policy
        self._Y = tf.compat.v1.placeholder(shape=[None, 6], dtype=tf.float32)  # 6 output cases
        
        # Error clip
        error = self.cliped_error(self._Y - self._Qpred)

        # Loss function
        self._loss = tf.reduce_mean(tf.square(error))
        
        # Learning
        self._train = tf.compat.v1.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=EPSILON).minimize(self._loss)
#        self._train = tf.compat.v1.train.AdamOptimizer(LEARNING_RATE).minimize(self._loss)
       
    def predict(self, state):
        return self.session.run(self._Qpred, feed_dict={self._X: state})
    
    def update(self, x_stack, y_stack):
        return self.session.run([self._loss, self._train], feed_dict={self._X: x_stack, self._Y: y_stack})  


def replay_train(mainDQN, targetDQN, minibatch):
    x_stack = np.empty(0).reshape(-1, 80,80,4) # array([], shape=(0, input_size), dtype=float64)
    y_stack = np.empty(0).reshape(-1, 6) # array([], shape=(0, output_size), dtype=float64)
    
    for state, action, reward, next_state, done in minibatch:
        Q = mainDQN.predict(state)
        
        if(done):
            Q[0, action] = reward
        else:
            Q[0, action] = reward + DISCOUNT_FACTOR * np.max(targetDQN.predict(next_state))

        x_stack = np.vstack([x_stack, state])
        y_stack = np.vstack([y_stack, Q])
    
    return mainDQN.update(x_stack, y_stack)


def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
    op_holder = []
    src_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
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
    
    # Staring parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test', action='store_true', help="test preprocessing")   
    parser.add_argument('-l', '--load', action='store_true', help="load model")
    parser.add_argument('-r', '--render', action='store_true', help="play trained model")

    args = parser.parse_args()

    test = args.test # default is False
    load = args.load # default is False
    render = args.render # default is False
    
    # test preprocessing
    if(test):
        env.reset()
        preprocess_test(env)
        sys.exit(0)
    
    # line 1 (initialize replay memory D to capacity N)
    replay_buffer = deque()
    
    with tf.compat.v1.Session() as sess:
#    sess = tf.InteractiveSession()
        mainDQN = DQN(sess, INPUT_H, INPUT_W, CHANNEL, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_H, INPUT_W, CHANNEL, OUTPUT_SIZE, name="target")
        tf.compat.v1.global_variables_initializer().run()
    
        # Add ops to save and restore all the variables.
        saver = tf.compat.v1.train.Saver()
        
        # Load trained parameters
        if(load):
            saver.restore(sess, MODEL_SAVE_LOCATION)
            print("model loaded.")
    
        # Render model
        if(render):
            bot_play(mainDQN)    
        
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        
        sess.run(copy_ops)
        
        loss = 0
        frame = 1
        
        for episode in range(TOTAL_EPISODE):
            # Exploration variable: [Initial = 1, Final = 0.1]
            
            if(frame > TRAIN_START & EXPLORATION > 0.1):
                EXPLORATION = 1 - 0.9 * ((frame-TRAIN_START) / FINAL_EXPLORATION_FRAME)

            done = False
            reward_total = 0
            
            # line 5-1 (Initialize sequence s_q = {x_1})
            state = env.reset()
    
            # line 5-2 (preprocessed sequenced phi_1 = phi(s_1))
            state = state2 = state3 = state4 = preprocess(state) # preprocess state (210, 160, 3) -> (80, 80)
    
            qlist = []
    
            # line 6 (for t = 1,T do)
            while not done: 
                
                # calculate Q value            
                tmp = np.concatenate((state, state2, state3, state4), axis=2).reshape(-1,80,80,4)
                q = mainDQN.predict(tmp)
                qmax = np.max(q)
                qlist.append(qmax)
                
                # line 7 (With probability e select a random action a_t)
                if(np.random.rand(1) < EXPLORATION):
                    action = env.action_space.sample()
                else: # line 8 (otherwise select a_t = max ...)
                    action = np.argmax(q)
                    
                # line 9 (execute action a_t in emulator and observe reward r_t and image x_t+1)                    
                next_state, reward, done, _ = env.step(action) # get next states, rewards, dones
                frame += 1

                # make it 4 frames for each state and next_state
                s_t = np.concatenate((state, state2, state3, state4), axis=2).reshape(-1,80,80,4)
                
                state = state2
                state2 = state3
                state3 = state4
                state4 = preprocess(next_state)
                
                s2_t = np.concatenate((state, state2, state3, state4), axis=2).reshape(-1,80,80,4)
                    
                # put elements into the buffer
                replay_buffer.append((s_t, action, reward, s2_t, done))
    
                # spit out elements if it has more than REPLAY_MEMORY_SIZE
                while(len(replay_buffer) > REPLAY_MEMORY_SIZE):
                    replay_buffer.popleft()
    
                # accumulate total reward
                reward_total += reward
    
                # update main DQN
                if(frame >= TRAIN_START):
                    minibatch = random.sample(replay_buffer, MINIBATCH_SIZE) # MINIBATCH_SIZE == 32
                    # Every C updates we clone the network Q to obtaion a target network Qhat
                    if(frame % TARGET_NETWORK_UPDATE_FREQUENCY == 0): # TARGET_NETWORK_UPDATE_FREQUENCY = 10000        
                        # copy network from mainDQN to targetDQN
                        sess.run(copy_ops)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    
        
                if(frame % (TARGET_NETWORK_UPDATE_FREQUENCY * 5) == 0): # save per 50000 frames
                    # save model
                    saver.save(sess, MODEL_SAVE_LOCATION)
                    print("model saved.")
                    
    
            print("Episode: {:5d}, frame: {:6d}, reward_total: {}, avgMaxQ: {:.4f}, e: {:.4f}, loss: {:.7f}".format(episode, frame, reward_total, np.mean(qlist), EXPLORATION, loss))
            
            # stop looping if total reward is bigger than 12
            if(reward_total > 12):
                print("congrats! reward above 12")
                break
