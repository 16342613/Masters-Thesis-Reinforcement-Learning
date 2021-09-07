import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter

from ReplayBuffer import ReplayBuffer
from StateTransition import StateTransition


class DeepQLearner:
    def __init__(self, discountFactor, batchSize, inputShape, actionCount):
        self.discountFactor = discountFactor
        self.batchSize = batchSize
        self.inputShape = inputShape
        self.actionCount = actionCount

        self.predictionNetwork = None
        self.targetNetwork = None

        if os.path.isfile("Generated Data/Saved Models/targetNetwork.h5"):
            self.targetNetwork = keras.models.load_model("Generated Data/Saved Models/targetNetwork.h5")
            print("Loaded target network")
        if os.path.isfile("Generated Data/Saved Models/predictionNetwork.h5"):
            self.predictionNetwork = keras.models.load_model("Generated Data/Saved Models/predictionNetwork.h5")
            print("Loaded prediction network")

        self.replayBuffer = ReplayBuffer("Generated Data/Replay Buffer/")
        self.currentRewards = []
        self.losses = []
        self.qValueDistribution = []

        self.parseMapping = dict({"UnityEngine.Vector3": self.__parse_vector3,
                                  "System.Single": self.__parse_float,
                                  "System.Int32": self.__parse_float  # Treat integers as floats
                                  })

    def set_up_networks(self):
        if self.predictionNetwork is None:
            self.predictionNetwork = self.create_neural_network()
        if self.targetNetwork is None:
            self.targetNetwork = self.create_neural_network()
            self.targetNetwork.set_weights(self.predictionNetwork.get_weights())

    def update_target_network(self, placeholderInput):
        self.targetNetwork.set_weights(self.predictionNetwork.get_weights())
        print(datetime.now().strftime(
            "%H:%M:%S") + " : Updated the target network________________________________________________________________________________________________")
        return "1"

    # def create_neural_network(self):
    #     model = keras.Sequential([
    #         layers.Dense(256, input_shape=self.inputShape, activation="relu"),
    #         layers.Dense(256, activation="relu"),
    #         layers.Dense(256, activation="relu"),
    #         layers.Dense(self.actionCount, activation="linear")
    #     ])
    #
    #     model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss=self.dql_loss)
    #
    #     return model

    def create_neural_network(self):
        inputs = Input(shape=self.inputShape)
        x = layers.Dense(24, activation="relu")(inputs)
        x = layers.Dense(24, activation="relu")(x)
        x = layers.Dense(self.actionCount, activation="linear")(x)

        model = Model(inputs, x)
        model.compile(loss="mse", optimizer=keras.optimizers.Adam(learning_rate=0.001))
        return model

    def predict_action(self, stringInput, providedState = None):
        if providedState is None:
            parsedInputs = self.parse_string_input(stringInput)
            predictions = self.predictionNetwork.predict(parsedInputs)

            stringOutput = ""
            for actionProbability in predictions[0]:
                stringOutput += str(actionProbability) + " | "

            return stringOutput[:-3]
        else:
            predictions = self.predictionNetwork.predict(providedState)
            return predictions

    def parse_string_input(self, stringInput, delimiter=" | "):
        types = stringInput.split(" >|< ")[0].split(delimiter)
        splitData = stringInput.split(" >|< ")[1].split(delimiter)
        parsedData = []

        for dataIndex in range(len(splitData)):
            parsedData.extend(self.parseMapping[types[dataIndex]](splitData[dataIndex]))

        return np.array(parsedData).reshape(1, len(parsedData))

    @staticmethod
    def __parse_vector3(inputString):
        stringData = inputString.replace("(", "").replace(")", "").split(", ")
        return [float(data) for data in stringData]

    @staticmethod
    def __parse_float(inputString):
        return [float(inputString)]

    def add_to_replay_buffer(self, stringInput, providedTransition=None):
        if providedTransition is None:
            splitString = stringInput.split(" >|< ")
            transitionData = StateTransition(self.parse_string_input(" >|< ".join([splitString[0], splitString[1]]))[0],
                                             self.__parse_float(splitString[2])[0],
                                             self.__parse_float(splitString[3])[0],
                                             self.parse_string_input(" >|< ".join([splitString[4], splitString[5]]))[0],
                                             int(self.__parse_float(splitString[6])[0]))

            self.replayBuffer.populate_buffer(transitionData)
        else:
            self.replayBuffer.populate_buffer(providedTransition)
        return str(len(self.replayBuffer.buffer))

    # def dql_loss(self, y_true, y_pred):
    #     lossSum = tf.cast(tf.constant(0), dtype=tf.float32)
    #
    #     for i in range(len(y_true)):
    #         actualY = y_true[i][0]
    #         predictedY = y_pred[i][0][self.chosenActions[i]]  # tf.reduce_max(y_pred[i][0])
    #         loss = tf.square(actualY - predictedY)
    #         lossSum = lossSum + loss
    #
    #     loss = lossSum / len(y_true)
    #     return loss

    def plot_progress(self, stringInput):
        splitString = stringInput.split(" | ")

        for i in range(len(splitString)):
            self.currentRewards.append(float(splitString[i]))

        x = [i for i in range(len(self.currentRewards))]

        #ySmooth = savgol_filter(self.currentRewards, 25, 3)
        #axes = plt.gca()
        #axes.set_ylim([0, 50])
        plt.plot(x, self.currentRewards, color='red')
        #plt.plot(x, ySmooth, color='black')
        if os.path.isfile("Generated Data/Screenshots/latestPlot.png"):
            os.remove("Generated Data/Screenshots/latestPlot.png")
        plt.savefig("Generated Data/Screenshots/latestPlot.png")
        plt.clf()

        x = [i for i in range(len(self.losses))]
        #axes = plt.gca()
        #axes.set_ylim([0, 50])
        plt.plot(x, self.losses, color='green')
        if os.path.isfile("Generated Data/Screenshots/latestLoss.png"):
            os.remove("Generated Data/Screenshots/latestLoss.png")
        plt.savefig("Generated Data/Screenshots/latestLoss.png")
        plt.clf()

        for i in range(self.qValueDistribution.shape[1]):
            # plt.plot([j for j in range(self.qValueDistribution.shape[0])], self.qValueDistribution[:, i], label = str(i))
            plt.plot([j for j in range(499)], self.qValueDistribution[(self.qValueDistribution.shape[0] - 500):(self.qValueDistribution.shape[0] - 1), i], label=str(i))
        plt.legend()
        if os.path.isfile("Generated Data/Screenshots/QValueDistribution.png"):
            os.remove("Generated Data/Screenshots/QValueDistribution.png")
        plt.savefig("Generated Data/Screenshots/QValueDistribution.png")
        plt.clf()

        print("Saved plots")

        return "1.0"

    def __compute_target_Q_value(self, newState, reward):
        #rawTarget = np.amax(self.targetNetwork.predict(np.array(newState).reshape(1, self.inputShape[0])))
        rawTarget = np.amax(self.predictionNetwork.predict(np.array(newState).reshape(1, self.inputShape[0])))
        return reward + (self.discountFactor * rawTarget)

    def train(self):
        # Sample from replay buffer
        sampledTransitions = self.replayBuffer.sample_buffer(self.batchSize)
        # sampledTransitions = self.replayBuffer.prioritized_experience_sample(self.batchSize, (0.5, 0.5))
        # print(len(self.replayBuffer.buffer))
        stateBatch = []
        qValuesBatch = []
        losses = []
        for transition in sampledTransitions:
            predictionQValues = self.predictionNetwork.predict(np.array(transition.initialState).reshape(1, self.inputShape[0]))
            self.__add_to_q_value_distribution(predictionQValues[0])
            if transition.terminalState == 1:
                targetQValue = transition.reward
            else:
                targetQValue = self.__compute_target_Q_value(transition.newState, transition.reward)
            predictionQValues[0][int(transition.action)] = targetQValue

            stateBatch.append(transition.initialState)
            qValuesBatch.append(predictionQValues[0])

            x = transition.initialState.reshape(1, 4)
            history = self.predictionNetwork.fit(x, predictionQValues, epochs=1, verbose=0)
            losses.append(history.history["loss"][0])
        print(round(sum(losses)/len(losses), 5))
        #history = self.predictionNetwork.fit(np.array(stateBatch), np.array(qValuesBatch), batch_size=self.batchSize, epochs=1)
        #self.losses.append(history.history["loss"][0])

        # trainX = np.array([[transition.initialState] for transition in sampledTransitions])
        # trainY = []
        #
        # self.chosenActions = []
        # predictionQ = []
        #
        # # Calculate the outputs of the target network, taking the reward and discount factor into account
        # for i in range(len(sampledTransitions)):
        #     predictionQValues = self.predictionNetwork.predict(sampledTransitions[i].initialState)
        #
        #     trainY.append(sampledTransitions[i].reward + (self.discountFactor * np.max(
        #         self.targetNetwork.predict(sampledTransitions[i].newState.reshape(1, self.inputShape[0])))))
        #     # trainY.append(sampledTransitions[i].reward + (self.discountFactor * self.targetNetwork.predict(sampledTransitions[i].newState.reshape(1, self.inputShape[0]))[0][int(sampledTransitions[i].action)]))
        #     self.chosenActions.append(int(sampledTransitions[i].action))
        #
        # trainY = np.array(trainY)
        # history = self.predictionNetwork.fit(trainX, trainY, epochs=1, batch_size=self.batchSize, verbose=1)
        # self.losses.append(history.history["loss"][0])

    def save_models(self):
        self.targetNetwork.save("Generated Data/Saved Models/targetNetwork.h5")
        self.predictionNetwork.save("Generated Data/Saved Models/predictionNetwork.h5")

    def update_reward(self, stringInput):
        inputs = self.parse_string_input(stringInput)[0]
        self.replayBuffer.update_reward(int(inputs[0]), inputs[1])

        return "0"

    def __add_to_q_value_distribution(self, predictionOutput):
        data = []

        if len(self.qValueDistribution) == 0:
            for qValue in predictionOutput:
                data.append(qValue / np.sum(predictionOutput))

            self.qValueDistribution.append(data)
            self.qValueDistribution = np.array(self.qValueDistribution)
        else:
            for qValue in predictionOutput:
                data.append(qValue / np.sum(predictionOutput))

            self.qValueDistribution = np.concatenate((self.qValueDistribution, [data]))




