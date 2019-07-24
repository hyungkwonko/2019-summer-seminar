# -*- coding: utf-8 -*-
"""
@source: https://www.youtube.com/watch?v=Fcmgl8ow2Uc&list=PLlMkM4tgfjnKsCWav-Z2F-MMFRx-2gMGG&index=12
@written by: Sung KIM
@modified by: Hyung-Kwon Ko
@created on: Wed Jul 24 15:54:36 2019
@last modified date: 2019-07-24
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# set parameters
learning_rate = 0.1
dis = 0.99
num_episodes = 2000

class Lab6:
    def __init__(self, id):
        self.env = gym.make(id)
        # size of input & output
        if(id == "FrozenLake-v0"):
            self.input_size = self.env.observation_space.n # input size = 16 (# of locations)
        elif(id == "CartPole-v0"):
            self.input_size = self.env.observation_space.shape[0]
        else:
            print("no id found error")
        self.output_size = self.env.action_space.n # output size = 4 (left, right, up, down)
        
    # generate one hot
    def one_hot(self, x):
        return np.identity(16)[x:x + 1]
    
    # set placeholder & variable
    def setTF(self, lr = 0.1):
        self.X = tf.placeholder(shape=[1,self.input_size], dtype=tf.float32) # input
        self.W = tf.Variable(tf.random_uniform([self.input_size, self.output_size], 0, 0.01)) # initialized w/ the numbers between (0 & 0.01)
        self.Yhat = tf.matmul(self.X, self.W) # y hat (prediction)
        self.Y = tf.placeholder(shape=[1, self.output_size], dtype=tf.float32) # y
        
        # loss function
        loss = tf.reduce_sum(tf.square(self.Y - self.Yhat))
        
        # training model using gradient descent optimizer
        self.train = tf.train.GradientDescentOptimizer(lr).minimize(loss)
        
    # run tensorflow model
    def runTF(self, iter = 2000, dis = 0.99):

        rList = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(iter):
                s = self.env.reset()
                e = 1. / ((i / 50) + 10)
                rAll = 0
                done = False
                
                while not done:
                    Qs = sess.run(self.Yhat, feed_dict={self.X: self.one_hot(s)})
                    if np.random.rand(1) < e:
                        a = self.env.action_space.sample()
                    else:
                        a = np.argmax(Qs)
                        
                    s1, reward, done, _ = self.env.step(a)
                    
                    if done:
                        Qs[0, a] = reward
                    else:
                        Qs1 = sess.run(self.Yhat, feed_dict = {self.X: self.one_hot(s1)})
                        Qs[0, a] = reward + dis * np.max(Qs1)
                    
                    sess.run(self.train, feed_dict = {self.X: self.one_hot(s), self.Y:Qs})
                    
                    rAll += reward
                    s = s1
                rList.append(rAll)
        return rList
        
    # set placeholder & variable
    def setTF2(self, lr = 0.1):
        self.X = tf.placeholder(tf.float32, [None, self.input_size], name = "input_x") # input
        self.W = tf.get_variable("W1", shape = [self.input_size, self.output_size],
                                 initializer = tf.contrib.layers.xavier_initializer())
        self.Yhat = tf.matmul(self.X, self.W) # y hat (prediction)
        self.Y = tf.placeholder(shape=[None, self.output_size], dtype=tf.float32) # y        

        # loss function
        loss = tf.reduce_sum(tf.square(self.Y - self.Yhat))
        
        # training model using Adam optimizer
        self.train = tf.train.AdamOptimizer(lr).minimize(loss)
    

    # run tensorflow model2
    def runTF2(self, iter = num_episodes, dis = 0.99):

        rList = []
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for i in range(iter):
                s = self.env.reset()
                e = 1. / ((i / 10) + 1)
                step_count = 0
                done = False
                
                while not done:
                    step_count += 1
                    x = np.reshape(s, [1, self.input_size])

                    Qs = sess.run(self.Yhat, feed_dict={self.X: x})
                    if np.random.rand(1) < e:
                        a = self.env.action_space.sample()
                    else:
                        a = np.argmax(Qs)
                        
                    s1, reward, done, _ = self.env.step(a)
                    
                    if done:
                        Qs[0, a] = -100
                    else:
                        X1 = np.reshape(s1, [1, self.input_size])
                        Qs1 = sess.run(self.Yhat, feed_dict = {self.X: X1})
                        Qs[0, a] = reward + dis * np.max(Qs1)
                    
                    sess.run(self.train, feed_dict = {self.X: x, self.Y:Qs})                    
                    s = s1

                rList.append(step_count)
                print("Episodes: {}, steps: {}".format(i, step_count))
                
                if(len(rList) > 10 and np.mean(rList[-10:]) > 500):
                    break

        observation = self.env.reset()
        reward_sum = 0
        while True:
            self.env.render()
    
            x = np.reshape(observation, [1, self.input_size])
            Qs = sess.run(self.Qpred, feed_dict={self.X: x})
            a = np.argmax(Qs)
    
            observation, reward, done, _ = self.env.step(a)
            reward_sum += reward
            if done:
                print("Total score: {}".format(reward_sum))
                break

        
# run program
if __name__ == "__main__":

    # First - FrozenLake
    l6_1 = Lab6('FrozenLake-v0')
    l6_1.setTF(learning_rate)
    rList = l6_1.runTF(iter = num_episodes, dis = dis)

    print("Percent of successful episodes: " + str(sum(rList) / num_episodes) + "%")
    plt.bar(range(len(rList)), rList, color = "blue")
    plt.show()

    # Second - CartPole
    l6_2 = Lab6('CartPole-v0')
    l6_2.setTF2(learning_rate)
    rList = l6_2.runTF2()

