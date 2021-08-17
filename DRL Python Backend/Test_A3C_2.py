import os
import threading
from queue import Queue
import gym
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from A3C_Buffer import A3C_Buffer
from A3C_Worker import A3C_Worker
from A3C_NN import A3C_NN
from StateTransition import StateTransition
import numpy as np
from Global import Global
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


class GamePlayer:
    def __init__(self, workerCount):
        self.workerCount = workerCount
        self.optimiser = tf.compat.v1.train.AdamOptimizer(0.00025, use_locking=True)
        self.workers = []
        self.stateSize = gym.make("CartPole-v0").unwrapped.observation_space.shape[0]
        self.actionSize = gym.make("CartPole-v0").unwrapped.action_space.n
        Global.globalModel = A3C_NN(self.stateSize, self.actionSize)
        Global.globalModel(tf.convert_to_tensor(np.random.random((1, self.stateSize)), dtype=tf.float32))

    def start(self):
        for i in range(self.workerCount):
            env = Environment(self.stateSize, self.actionSize, self.optimiser)
            self.workers.append(env)
            self.workers[i].start()


class Environment(threading.Thread):
    def __init__(self, stateSize, actionSize, optimiser):
        super(Environment, self).__init__()
        self.env = gym.make("CartPole-v0").unwrapped
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.local_model = A3C_NN(stateSize, actionSize)
        self.episodeCount = 0
        self.worker = A3C_Worker(stateSize, actionSize, 1, 0.99, optimiser, weightUpdateInterval=15)
        self.rewards = []

    def run(self):
        while self.episodeCount < 500:
            currentState = self.env.reset()
            self.worker.memory.clear_buffer()
            episodeReward = 0

            done = 0
            while done == 0:
                logits, _ = self.local_model(tf.convert_to_tensor(currentState[None, :],
                                                                  dtype=tf.float32))
                probabilities = tf.nn.softmax(logits)

                action = np.random.choice(self.actionSize, p=probabilities.numpy()[0])
                newState, reward, doneBool, _ = self.env.step(action)

                if doneBool is True:
                    done = 1
                else:
                    done = 0

                episodeReward += reward
                self.worker.memory.populate_buffer(StateTransition(currentState, action, reward, newState, done))
                self.worker.append_to_buffer("", True)

                if done == 1 and len(self.worker.memory.buffer) > 0:
                    episodeReward -= 1
                    self.worker.train()

            self.episodeCount += 1
            print(self.episodeCount)
            self.rewards.append(episodeReward)

            if episodeReward > Global.bestReward:
                Global.bestReward = episodeReward
                print(" --> " + str(Global.bestReward) + " is now the best reward!")

        ySmooth = savgol_filter(self.rewards, 99, 6)
        plt.plot([i for i in range(len(self.rewards))], self.rewards)
        plt.plot([i for i in range(len(self.rewards))], ySmooth, color='black')
        plt.show()


player = GamePlayer(12)
player.start()
