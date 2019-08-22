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
REPLAY_MEMORY_SIZE = 50000 # 500000
AGENT_HISTORY_LENGTH = 4
TARGET_NETWORK_UPDATE_FREQUENCY = 1000 # 10000
TRAIN_START = 1000
DISCOUNT_FACTOR = 0.95 # 0.99
ACTION_REPEAT = 4
UPDATE_FREQUENCY = 4 # ???
LEARNING_RATE = 0.025 # 0.00025
EXPLORATION = 1 # initial value
FINAL_EXPLORATION_FRAME = 10000 # 1000000 Should be run after this number, but set it as our final number of steps to run
TOTAL_NUM_EPISODE = FINAL_EXPLORATION_FRAME * 10
MODEL_SAVE_LOCATION = "c:/users/sunbl/desktop/model/model.ckpt"
VIDEO_SAVE_LOCATION = "c:/users/sunbl/desktop/pong"

# Used Deterministic-v4 (action is selected for every 4 frames)
# https://github.com/openai/gym/blob/5cb12296274020db9bb6378ce54276b31e7002da/gym/envs/__init__.py#L352
env = gym.make("PongDeterministic-v4")
#env = gym.make("MontezumaRevengeDeterministic-v4")

# record the game as as an mp4 file
# how to use: https://gym.openai.com/evaluations/eval_lqShqslRtaJqR9yWWZIJg/
env = wrappers.Monitor(env, VIDEO_SAVE_LOCATION,  force=True, video_callable=lambda episode_id: episode_id%TARGET_NETWORK_UPDATE_FREQUENCY==0) 

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
    
    def _build_network(self, l_rate=LEARNING_RATE):
        with tf.variable_scope(self.net_name):

            # input, SIZE: (batch_size, 80, 80, 4)
            self._X = tf.placeholder(tf.float32, [None,80,80,4])

            # layer 1
            # SIZE: 80x80, CHANNEL: 4, FILTER: 16 (8x8)
            W1 = tf.Variable(tf.random_normal([8,8,4,16], stddev=0.1), name="W1")

            # STRIDE: 4x4, PADDING: same
            L1 = tf.nn.conv2d(self._X, W1, strides=[1,4,4,1], padding='SAME')
            L1 = tf.nn.relu(L1)
            
            # layer 2
            # SIZE: 20x20, CHANNEL: 16, FILTER: 32 (4x4)
            W2 = tf.Variable(tf.random_normal([4,4,16,32], stddev = 0.1), name="W2")
            
            # STRIDE: 2x2, PADDING: same
            L2 = tf.nn.conv2d(L1, W2, strides=[1,2,2,1], padding='SAME')
            L2 = tf.nn.relu(L2)
            
            # SIZE: 10x10, CHANNEL: 32, 10x10x32= 3200
            L2 = tf.reshape(L2, [-1,3200])
            
            # fully-connected layer: 3200 -> 256
            W3 = tf.get_variable("W3", shape=[3200, 256], initializer=tf.contrib.layers.xavier_initializer())            
            L3 = tf.nn.tanh(tf.matmul(L2, W3))
            
            # output layer, 256 -> 6
            W4 = tf.get_variable("W4", shape=[256, 6], initializer=tf.contrib.layers.xavier_initializer())        
            self._Qpred = tf.matmul(L3, W4)
        
        # Policy
        self._Y = tf.placeholder(shape=[None, 6], dtype=tf.float32)  # 6가지의 output cases

        # Loss function
        self._loss = tf.reduce_mean(tf.square(self._Y - self._Qpred))
        
        # Learning
        self._train = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(self._loss)
        
    def predict(self, state):
        x = state.reshape([-1, 80,80,4])
        return self.session.run(self._Qpred, feed_dict={self._X: x})
    
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
    
    with tf.Session() as sess:
#    sess = tf.InteractiveSession()
        mainDQN = DQN(sess, INPUT_H, INPUT_W, CHANNEL, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_H, INPUT_W, CHANNEL, OUTPUT_SIZE, name="target")
        tf.global_variables_initializer().run()
    
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()
        
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
        
        for episode in range(TOTAL_NUM_EPISODE):
            # Exploration variable: [Initial = 1, Final = 0.1]
            
            if(episode > TRAIN_START):
                EXPLORATION = 1 - 0.9 * (episode / FINAL_EXPLORATION_FRAME)

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
                done = done or done2 or done3 or done4
    
                # preprocess next_state also
                next_state = preprocess(next_state)
                next_state2 = preprocess(next_state2)
                next_state3 = preprocess(next_state3)
                next_state4 = preprocess(next_state4)
                
                # make it 4 frames for each state and next_state
                s_t = np.concatenate((state, state2, state3, state4), axis=2).reshape(-1,80,80,4)
                s2_t = np.concatenate((next_state, next_state2, next_state3, next_state4), axis=2).reshape(-1,80,80,4)
                
                # sum of reward
                reward_t = reward + reward2 + reward3 + reward4
    
                # put elements into the buffer
                replay_buffer.append((s_t, action, reward_t, s2_t, done))
    
                # spit out elements if it has more than REPLAY_MEMORY_SIZE
                while(len(replay_buffer) > REPLAY_MEMORY_SIZE):
                    replay_buffer.popleft()
    
                # update state
                state = next_state
                state2 = next_state2
                state3 = next_state3
                state4 = next_state4
    
                # accumulate total reward
                reward_total = reward_total + reward_t
    
                # update main DQN
        #        for _ in range(UPDATE_FREQUENCY): # UPDATE_FREQUENCY == 4
                if(episode >= TRAIN_START):
                    minibatch = random.sample(replay_buffer, MINIBATCH_SIZE) # MINIBATCH_SIZE == 32
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    
            if(episode % TARGET_NETWORK_UPDATE_FREQUENCY == 1): # TARGET_NETWORK_UPDATE_FREQUENCY should be 10000

                # copy network from mainDQN to targetDQN
                sess.run(copy_ops)
    
                # save model
                saver.save(sess, MODEL_SAVE_LOCATION)
                print("model saved.")
                
    
            print("Episode: {}, reward_total: {}, avgMaxQ: {:.4f}, e: {:.4f}, loss: {}".format(episode, reward_total, np.mean(qlist), EXPLORATION, loss))
            
            # stop looping if total reward is bigger than 10
            if(reward_total > 10):
                break
    

