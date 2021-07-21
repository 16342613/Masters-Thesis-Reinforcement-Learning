import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import numpy as np
from tensorflow import keras

from ReplayBuffer import ReplayBuffer
from StateTransition import StateTransition


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
        self.replayBuffer = ReplayBuffer("Generated Data/")

        self.parseMapping = dict({"UnityEngine.Vector3": self.__parse_vector3,
                                  "System.Single": self.__parse_float,
                                  "System.Int32": self.__parse_float  # Treat integers as floats
                                  })

    def set_up_networks(self):
        self.predictionNetwork = self.create_neural_network()
        self.targetNetwork = self.create_neural_network()
        self.targetNetwork.set_weights(self.predictionNetwork.get_weights())


    def update_target_network(self):
        self.targetNetwork.set_weights(self.predictionNetwork.get_weights())


    def create_neural_network(self):
        model = keras.Sequential([
            layers.Dense(10, input_shape=self.inputShape, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(10, activation="relu"),
            layers.Dense(self.actionCount, activation="relu")
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


    def train(self):
        # Sample from replay buffer
        sampledTransitions = self.replayBuffer.sample_buffer(self.batchSize)
        trainX = np.array([[transition.initialState] for transition in sampledTransitions])
        trainY = []

        # Calculate the outputs of the target network, taking the reward and discount factor into account
        for i in range(len(sampledTransitions)):
            trainY.append(sampledTransitions[i].reward + (self.discountFactor * np.max(self.targetNetwork.predict(sampledTransitions[i].newState.reshape(1, self.inputShape[0])))))

        trainY = np.array(trainY)
        self.predictionNetwork.fit(trainX, trainY, epochs=10)


