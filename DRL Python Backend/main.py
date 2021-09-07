# import os
# import tensorflow as tf
#
# projectDirectory = r"E:\Users\mandh\Masters-Thesis-Reinforcemenet-Learning\Tanks DRL"
# replayBufferPath = "Assets/API Entry/AI/Replay Buffer.txt"
# os.chdir(projectDirectory)
# print(os.getcwd())
#
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# import os
#
# # Disable tensorflow warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
#
# from DeepQLearner import DeepQLearner
# from Server import Server
#
# dql = DeepQLearner(0.99, 64, (7,), 3)
# dql.set_up_networks()
#
# server = Server("DRL Backend", 8000, verboseLogging=False)
# server.set_deep_q_learner(dql)
# server.initialise_server()
# #server.start_server()
#
#
# # message = "PREDICT >|< UnityEngine.Vector3 | UnityEngine.Vector3 | UnityEngine.Vector3 | System.Int32 | UnityEngine.Vector3 | System.Single | UnityEngine.Vector3 | System.Single >|< (5.0, 0.6, 5.2) | (0.0, 0.0, 0.0) | (0.0, 0.0, 0.0) | 260 | (8.6, 0.8, 7.2) | 175 | (0.5, 0.2, 0.5) | 100"
# # splitMessage = message.split(" >|< ")
# # x = server.commands[splitMessage[0]](" >|< ".join([splitMessage[i + 1] for i in range(len(splitMessage) - 1)]))
# # print(x)

import gym
import random
import os
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras import backend as K
import numpy as np

class Agent():
    def __init__(self, state_size, action_size):
        self.weight_backup = "cartpole_weight.h5"
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.learning_rate = 0.001
        self.gamma = 0.95
        self.exploration_rate = 1.0
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.brain = self._build_model()
        self.latestLoss = -1

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer="adam")
        K.set_value(model.optimizer.learning_rate, self.learning_rate)
        print("Learning rate before first fit:", model.optimizer.learning_rate.numpy())

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
        losses = []
        for state, action, reward, next_state, done in sample_batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.brain.predict(next_state)[0])
            target_f = self.brain.predict(state)
            target_f[0][action] = target
            history = self.brain.fit(state, target_f, epochs=1, verbose=0)
            losses.append(history.history["loss"][0])
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

        self.latestLoss = round(sum(losses)/len(losses), 5)


class CartPole:
    def __init__(self):
        self.sample_batch_size = 32
        self.episodes = 10000
        self.env = gym.make('CartPole-v1')

        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        self.agent = Agent(self.state_size, self.action_size)

    def run(self):
        try:
            for index_episode in range(self.episodes):
                state = self.env.reset()
                state = np.reshape(state, [1, self.state_size])

                done = False
                index = 0
                while not done:
                    self.env.render()

                    action = self.agent.act(state)

                    next_state, reward, done, _ = self.env.step(action)
                    next_state = np.reshape(next_state, [1, self.state_size])
                    self.agent.remember(state, action, reward, next_state, done)
                    state = next_state
                    index += 1
                print("Episode: " + str(index) + " ; Score: " + str())
                print(self.agent.exploration_rate)
                self.agent.replay(self.sample_batch_size)
        finally:
            self.agent.save_model()


if __name__ == "__main__":
    cartpole = CartPole()
    cartpole.run()
