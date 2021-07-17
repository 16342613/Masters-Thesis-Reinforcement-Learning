import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
from ReplayBuffer import ReplayBuffer
from tensorflow import keras


class DeepQLearner:
    def __init__(self, discountFactor, epsilon, epsilonBounds, batchSize, inputShape, actionCount):
        self.discountFactor = discountFactor
        self.epsilon = epsilon
        self.epsilonBounds = epsilonBounds
        self.batchSize = batchSize
        self.inputShape = inputShape
        self.actionCount = actionCount

        self.predictionNetwork = None
        self.targetNetwork = None
        self.replayBuffer = None

        self.parseMapping = dict({"UnityEngine.Vector3": self.__parse_vector3,
                                  "System.Single": self.__parse_float})


    def sample_replay_buffer(self):
        pass


    def create_neural_network(self):
        # Create the neural network

        # inputLayer = layers.Input(self.inputShape)
        #
        # hiddenLayers = layers.Dense(10, activation="relu")(inputLayer)
        # hiddenLayers = layers.Dense(10, activation="relu")(hiddenLayers)
        # hiddenLayers = layers.Dense(10, activation="relu")(hiddenLayers)
        #
        # outputLayer = layers.Dense(self.actionCount, activation="linear")(hiddenLayers)
        #
        # return Model(inputs=inputLayer, outputs=outputLayer)

        model = keras.Sequential([
            layers.Dense(10, input_shape=self.inputShape, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(self.actionCount, activation="sigmoid")
        ])

        model.compile(optimizer="adam", loss="Huber", metrics=['accuracy'])

        return model



    def start_learning(self):
        self.predictionNetwork = self.create_neural_network()
        self.targetNetwork = self.create_neural_network()


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

