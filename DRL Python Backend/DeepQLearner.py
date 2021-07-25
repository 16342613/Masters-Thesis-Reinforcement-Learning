import os
from datetime import datetime
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
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
            self.targetNetwork = keras.models.load_model("Generated Data/Saved Models/targetNetwork.h5", custom_objects={'dql_loss': self.dql_loss})
            print("Loaded target network")
        if os.path.isfile("Generated Data/Saved Models/predictionNetwork.h5"):
            self.predictionNetwork = keras.models.load_model("Generated Data/Saved Models/predictionNetwork.h5", custom_objects={'dql_loss': self.dql_loss})
            print("Loaded prediction network")

        self.replayBuffer = ReplayBuffer("Generated Data/Replay Buffer/")
        self.currentRewards = []

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
        print(datetime.now().strftime("%H:%M:%S") + " : Updated the target network________________________________________________________________________________________________")
        return "1"


    def create_neural_network(self):
        model = keras.Sequential([
            layers.Dense(32, input_shape=self.inputShape, activation="relu", kernel_regularizer="l2"),
            layers.Dense(32, activation="relu", kernel_regularizer="l2"),
            layers.Dense(32, activation="relu", kernel_regularizer="l2"),
            layers.Dense(32, activation="relu", kernel_regularizer="l2"),
            layers.Dense(32, activation="relu", kernel_regularizer="l2"),
            layers.Dense(self.actionCount, activation="LeakyReLU", kernel_regularizer="l2")
        ])

        model.compile(optimizer="adam", loss=self.dql_loss, metrics=['accuracy'], run_eagerly=True)

        return model


    def predict_action(self, stringInput):
        parsedInputs = self.parse_string_input(stringInput)
        predictions = self.predictionNetwork.predict(parsedInputs)

        stringOutput = ""
        for actionProbability in predictions[0]:
            stringOutput += str(actionProbability) + " | "

        return stringOutput[:-3]

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

    def add_to_replay_buffer(self, stringInput):
        splitString = stringInput.split(" >|< ")
        transitionData = StateTransition(self.parse_string_input(" >|< ".join([splitString[0], splitString[1]]))[0],
                                         self.__parse_float(splitString[2])[0],
                                         self.__parse_float(splitString[3])[0],
                                         self.parse_string_input(" >|< ".join([splitString[4], splitString[5]]))[0])

        self.replayBuffer.populate_buffer(transitionData)
        return str(len(self.replayBuffer.buffer))

    def dql_loss(self, y_true, y_pred):
        lossSum = tf.cast(tf.constant(0), dtype=tf.float32)

        for i in range(len(y_true)):
            actualY = y_true[i][0]
            predictedY = tf.reduce_max(y_pred[i][0])
            loss = tf.square(actualY - predictedY)
            lossSum = lossSum + loss

        loss = lossSum / len(y_true)
        return loss


    def plot_progress(self, stringInput):
        splitString = stringInput.split(" | ")

        for i in range(len(splitString)):
            self.currentRewards.append(float(splitString[i]))

        x = [i for i in range(len(self.currentRewards))]

        ySmooth = savgol_filter(self.currentRewards, 25, 3)
        plt.plot(x, self.currentRewards, color='red')
        plt.plot(x, ySmooth, color='black')
        savePath = "Generated Data/Screenshots/latestPlot.png"
        if os.path.isfile(savePath):
            os.remove(savePath)
        plt.savefig(savePath)
        print("Saved plot")

        return "1.0"


    def train(self):
        # Sample from replay buffer
        sampledTransitions = self.replayBuffer.sample_buffer(self.batchSize)
        trainX = np.array([[transition.initialState] for transition in sampledTransitions])
        trainY = []
        qValues = []

        # Calculate the outputs of the target network, taking the reward and discount factor into account
        for i in range(len(sampledTransitions)):
            trainY.append(sampledTransitions[i].reward + (self.discountFactor * np.max(self.targetNetwork.predict(sampledTransitions[i].newState.reshape(1, self.inputShape[0])))))
            qValues.append(np.max(self.targetNetwork.predict(sampledTransitions[i].newState.reshape(1, self.inputShape[0]))))

        print("Avg Q values : " + str(sum(qValues) / len(qValues)))
        trainY = np.array(trainY)
        self.predictionNetwork.fit(trainX, trainY, epochs=25, batch_size=self.batchSize, verbose=1)

    def save_models(self):
        pass
        self.targetNetwork.save("Generated Data/Saved Models/targetNetwork.h5")
        self.predictionNetwork.save("Generated Data/Saved Models/predictionNetwork.h5")



