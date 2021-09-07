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
import Test_A3C


class GamePlayer:
    def __init__(self, workerCount):
        self.workerCount = workerCount
        self.optimiser = tf.compat.v1.train.AdamOptimizer(0.00025, use_locking=True)

        self.workers = []
        self.stateSize = 4
        self.actionSize = 2
        Global.globalModel = Test_A3C.ActorCriticModel(self.stateSize, self.actionSize)
        Global.globalModel(tf.convert_to_tensor(np.random.random((1, self.stateSize)), dtype=tf.float32))

        self.global_model = Test_A3C.ActorCriticModel(self.stateSize, self.actionSize)  # global network
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.stateSize)), dtype=tf.float32))

    def start(self):
        q = Queue()
        for i in range(self.workerCount):
            # env = Test_A3C.Worker(self.stateSize,
            #                       self.actionSize,
            #                       self.global_model,
            #                       self.optimiser, q,
            #                       i, game_name="CartPole-v0",
            #                       save_dir="Generated Data/Saved Models")  # Environment(self.stateSize, self.actionSize, self.optimiser)
            env = Environment(self.stateSize, self.actionSize, self.optimiser, self.global_model)
            self.workers.append(env)
            self.workers[i].start()


class Environment(threading.Thread):
    def __init__(self, stateSize, actionSize, optimiser, globalModel):
        super(Environment, self).__init__()
        self.env = gym.make("CartPole-v0").unwrapped
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.local_model = Test_A3C.ActorCriticModel(stateSize, actionSize)
        self.episodeCount = 0
        self.worker = A3C_Worker(stateSize, actionSize, 10, 0.99, optimiser, globalModel, weightUpdateInterval=15)
        self.rewards = []

    def run(self):
        while self.episodeCount < 4000:
            currentState = self.env.reset()
            self.worker.memory.clear_buffer()
            episodeReward = 0
            done = 0
            while done == 0:
                logits, _ = self.worker.localModel(tf.convert_to_tensor(currentState[None, :],
                                                                  dtype=tf.float32))
                probabilities = tf.nn.softmax(logits)

                action = np.random.choice(self.actionSize, p=probabilities.numpy()[0])
                newState, reward, doneBool, _ = self.env.step(action)

                if doneBool is True:
                    done = 1
                    reward = -1

                episodeReward += reward

                self.worker.memory.populate_buffer(StateTransition(currentState, action, reward, newState, ID=10, terminalState=done))
                self.worker.append_to_buffer("", True)

                currentState = newState

            self.episodeCount += 1
            print(str(self.episodeCount) + " -> " + str(episodeReward))
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
