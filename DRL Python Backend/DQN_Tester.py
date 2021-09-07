# import os
# from collections import deque
#
# import gym
# import numpy as np
# import random
# from scipy.signal import savgol_filter
# import matplotlib.pyplot as plt
#
# from DeepQLearner import DeepQLearner
# from StateTransition import StateTransition
#
# class DQN_Tester:
#     def __init__(self, gameName, maxEpisodes, trainingInterval, targetNetworkUpdateInterval):
#         self.gameName = gameName
#
#         self.env = gym.make(self.gameName)
#         self.stateSize = self.env.observation_space.shape[0]
#         self.actionSize = self.env.action_space.n
#
#         self.episodeCount = 0
#         self.maxEpisodes = maxEpisodes
#         self.trainingInterval = trainingInterval
#         self.targetNetworkUpdateInterval = targetNetworkUpdateInterval
#
#         self.deepQLearner = DeepQLearner(discountFactor=0.95, batchSize=32, inputShape=(self.stateSize,), actionCount=self.actionSize)
#         self.deepQLearner.set_up_networks()
#         self.bestReward = 0
#         self.epsilon = 1
#         self.episodeRewards = []
#
#
#     def play(self):
#         # Populate the replay buffer before training
#         for i in range(100):
#             episodeDone = False
#             currentState = self.env.reset()
#             while episodeDone is False:
#                 action = random.randint(0, self.actionSize - 1)
#                 newState, reward, episodeDone, _ = self.env.step(action)
#
#                 doneInt = 0
#                 if episodeDone:
#                     doneInt = 1
#                     reward = -1
#                     print("Completed population episode " + str(i))
#
#                 self.deepQLearner.add_to_replay_buffer("", StateTransition(currentState, action, reward, newState, 10, doneInt))
#
#         # Train the DQN
#         for i in range(self.maxEpisodes):
#             episodeDone = False
#             currentState = self.env.reset()
#             episodeReward = 0
#             trained = False
#             updatedTargetNetwork = False
#             episodeSteps = 0
#
#             while episodeDone is False:
#                 # self.env.render()
#                 if random.uniform(0, 1) < self.epsilon:
#                     action = random.randint(0, self.actionSize - 1)
#                 else:
#                     parsedState = currentState.reshape(1, self.stateSize)
#                     outputs = self.deepQLearner.predict_action("", providedState=parsedState)
#                     action = np.argmax(outputs[0])
#
#                 newState, reward, episodeDone, _ = self.env.step(action)
#
#                 doneInt = 0
#                 if episodeDone:
#                     doneInt = 1
#                     # reward = -1
#
#                     if self.epsilon > 0.05:
#                         self.epsilon = self.epsilon * 0.995
#                     print("Episode: " + str(i) + " ; Epsilon: " + str(round(self.epsilon, 3)) + " ; Reward: " + str(episodeReward))
#                     self.deepQLearner.train()
#                     self.episodeRewards.append(episodeReward)
#
#                     if i % 100 == 0 and i > 0:
#                         self.plot_data()
#
#                 episodeReward += reward
#                 self.deepQLearner.add_to_replay_buffer("", StateTransition(currentState, action, reward, newState, 10,
#                                                                            doneInt))
#
#                 if (i % self.trainingInterval == 0) and (i > 0) and (trained is False):
#                     # self.deepQLearner.train()
#                     trained = True
#
#                 if (i % self.targetNetworkUpdateInterval == 0) and (i > 0) and (updatedTargetNetwork is False):
#                     self.deepQLearner.update_target_network("")
#                     updatedTargetNetwork = True
#
#                 currentState = newState
#                 episodeSteps += 1
#
#                 if episodeDone is True and episodeReward > self.bestReward:
#                     print(str(episodeReward) + " is the new best reward!")
#                     self.bestReward = episodeReward
#
#     def plot_data(self):
#         ySmooth = savgol_filter(self.episodeRewards, 51, 6)
#         plt.plot([i for i in range(len(self.episodeRewards))], ySmooth, color='black')
#
#         plt.xlabel("Episode")
#         plt.ylabel("Reward")
#         plt.title("DQN Episode vs Reward")
#         # plt.show()
#
#         if os.path.isfile("Generated Data/Screenshots/DQN Plot.png"):
#             os.remove("Generated Data/Screenshots/DQN Plot.png")
#         plt.savefig("Generated Data/Screenshots/DQN Plot.png")
#
#
# tester = DQN_Tester("CartPole-v0", 1000, 1, 10000)
# tester.play()
#
#

# encoding: utf-8

##
## cartpole.py
## Gaetan JUVIN 06/24/2017
##

import gym
import random
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.0001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))

        if os.path.isfile(self.weight_backup):
            model.load_weights(self.weight_backup)
            self.exploration_rate = self.exploration_min
        return model

    def save_model(self):
        self.brain.save(self.weight_backup)

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        act_values = self.brain.predict(state)
        return np.argmax(act_values[0])

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            self.brain.fit(state, target_f, epochs=1, verbose=0)
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 2500
        self.env = gym.make('MountainCar-v0')
        self.env._max_episode_steps = 250

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)
        self.episodeRewards = []

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                episodeReward = 0
                while not done:
                    # self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                    episodeReward += reward
                # print("Episode {}# Score: {}".format(index_episode, index + 1))
                print("Episode: " + str(index_episode) + " ; Score: " + str(episodeReward))
                self.agent.replay(self.sample_batch_size)
                self.episodeRewards.append(episodeReward)
        finally:
            self.agent.save_model()
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


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
