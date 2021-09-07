import os

import gym
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import numpy as np
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

#############################
# if you want to use GPU to boost, use these code.

# import tensorflow as tf
# import keras
# config = tf.compat.v1.ConfigProto( device_count = {'GPU': 2 , 'CPU': 1} )
# sess = tf.compat.v1.Session(config=config)
# keras.backend.set_session(sess)

#############################


class MountainCarTrain:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.99

        self.epsilon = 1
        self.epsilon_decay = 0.05

        self.epsilon_min = 0.01

        self.learingRate = 0.001

        self.replayBuffer = deque(maxlen=20000)
        self.trainNetwork = self.createNetwork()

        self.episodeNum = 1000

        self.iterationNum = 201  # max is 200

        self.numPickFromBuffer = 32

        self.targetNetwork = self.createNetwork()

        self.targetNetwork.set_weights(self.trainNetwork.get_weights())
        self.episodeRewards = []

    def createNetwork(self):
        model = models.Sequential()
        state_shape = self.env.observation_space.shape

        model.add(layers.Dense(24, activation='relu', input_shape=state_shape))
        model.add(layers.Dense(48, activation='relu'))
        model.add(layers.Dense(self.env.action_space.n, activation='linear'))
        # model.compile(optimizer=optimizers.RMSprop(lr=self.learingRate), loss=losses.mean_squared_error)
        model.compile(loss='mse', optimizer=Adam(lr=self.learingRate))
        return model

    def getBestAction(self, state):

        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 3)
        else:
            action = np.argmax(self.trainNetwork.predict(state)[0])

        return action

    def trainFromBuffer_Boost(self):
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return
        samples = random.sample(self.replayBuffer, self.numPickFromBuffer)
        npsamples = np.array(samples)
        states_temp, actions_temp, rewards_temp, newstates_temp, dones_temp = np.hsplit(npsamples, 5)
        states = np.concatenate((np.squeeze(states_temp[:])), axis=0)
        rewards = rewards_temp.reshape(self.numPickFromBuffer, ).astype(float)
        targets = self.trainNetwork.predict(states)
        newstates = np.concatenate(np.concatenate(newstates_temp))
        dones = np.concatenate(dones_temp).astype(bool)
        notdones = ~dones
        notdones = notdones.astype(float)
        dones = dones.astype(float)
        Q_futures = self.targetNetwork.predict(newstates).max(axis=1)
        targets[(np.arange(self.numPickFromBuffer),
                 actions_temp.reshape(self.numPickFromBuffer, ).astype(int))] = rewards * dones + (
                rewards + Q_futures * self.gamma) * notdones
        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

    def trainFromBuffer(self):
        if len(self.replayBuffer) < self.numPickFromBuffer:
            return

        samples = random.sample(self.replayBuffer, self.numPickFromBuffer)

        states = []
        newStates = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            newStates.append(new_state)

        newArray = np.array(states)
        states = newArray.reshape(self.numPickFromBuffer, 2)

        newArray2 = np.array(newStates)
        newStates = newArray2.reshape(self.numPickFromBuffer, 2)

        targets = self.trainNetwork.predict(states)
        new_state_targets = self.targetNetwork.predict(newStates)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.gamma
            i += 1

        self.trainNetwork.fit(states, targets, epochs=1, verbose=0)

    def originalTry(self, currentState, eps):
        rewardSum = 0
        max_position = -99
        stepCount = 0
        for i in range(self.iterationNum):
            bestAction = self.getBestAction(currentState)

            # show the animation every 50 eps
            if eps % 50 == 0:
                env.render()

            new_state, reward, done, _ = env.step(bestAction)

            new_state = new_state.reshape(1, 2)

            # # Keep track of max position
            if new_state[0][0] > max_position:
                max_position = new_state[0][0]

            # # Adjust reward for task completion
            if new_state[0][0] >= 0.5:
                print(done)
                reward += 10

            self.replayBuffer.append([currentState, bestAction, reward, new_state, done])

            # Or you can use self.trainFromBuffer_Boost(), it is a matrix wise version for boosting
            # self.trainFromBuffer()
            self.trainFromBuffer_Boost()

            rewardSum += reward

            currentState = new_state
            stepCount += 1

            if done:
                break

        if stepCount >= 199:
            self.episodeRewards.append(rewardSum)
            print("Failed to finish task in epsoide {}".format(eps))
        else:
            self.episodeRewards.append(rewardSum)
            print("Success in epsoide {}, used {} iterations!".format(eps, stepCount))
            self.trainNetwork.save('./trainNetworkInEPS{}.h5'.format(eps))

        # Sync
        self.targetNetwork.set_weights(self.trainNetwork.get_weights())

        print("now epsilon is {}, the reward is {} maxPosition is {}".format(max(self.epsilon_min, self.epsilon),
                                                                             rewardSum, max_position))
        self.epsilon -= self.epsilon_decay

    def start(self):
        for eps in range(self.episodeNum):
            currentState = env.reset().reshape(1, 2)
            self.originalTry(currentState, eps)
        self.plot_data()

    def plot_data(self):
        ySmooth = savgol_filter(self.episodeRewards, 51, 6)
        plt.plot([i for i in range(len(self.episodeRewards))], ySmooth, color='black')

        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.title("DQN Episode vs Reward")
        # plt.show()

        if os.path.isfile("Generated Data/Screenshots/DQN Plot.png"):
            os.remove("Generated Data/Screenshots/DQN Plot.png")
        plt.savefig("Generated Data/Screenshots/DQN Plot.png")


# env = gym.make('MountainCar-v0')
#
# # play 20 times
# # load the network
# model = models.load_model('trainNetworkInEPS399.h5')
#
# for i_episode in range(20):
#     currentState1 = env.reset().reshape(1, 2)
#
#     print("============================================")
#
#     rewardSum = 0
#     for t in range(200):
#         env.render()
#         action = np.argmax(model.predict(currentState1)[0])
#
#         new_state, reward, done, info = env.step(action)
#
#         new_state = new_state.reshape(1, 2)
#
#         currentState1 = new_state
#
#         rewardSum += reward
#         if done:
#             print("Episode finished after {} timesteps reward is {}".format(t + 1, rewardSum))
#             break

env = gym.make('MountainCar-v0')
dqn = MountainCarTrain(env=env)
dqn.start()
