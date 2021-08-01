from A3C_NN import A3C_NN
from A3C_Buffer import A3C_Buffer
from StateTransition import StateTransition

import numpy as np
import tensorflow as tf
import threading
import time


class A3C_Worker(threading.Thread):
    def __init__(self, inputSize, actionCount, globalModel, workerID, discountFactor, optimiser, weightUpdateInterval = 50):
        super(A3C_Worker, self).__init__()
        self.inputSize = inputSize
        self.actionCount = actionCount
        self.globalModel = globalModel
        self.workerID = workerID
        self.discountFactor = discountFactor
        self.optimiser = optimiser

        self.parseMapping = dict({"UnityEngine.Vector3": self.__parse_vector3,
                                  "System.Single": self.__parse_float,
                                  "System.Int32": self.__parse_float  # Treat integers as floats
                                  })

        self.localModel = A3C_NN(self.inputSize, self.actionCount)
        self.memory = A3C_Buffer()

        self.weightUpdateInterval = weightUpdateInterval

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

    def append_to_buffer(self, stringInput):
        splitString = stringInput.split(" >|< ")
        transitionData = StateTransition(self.parse_string_input(" >|< ".join([splitString[0], splitString[1]]))[0],
                                         self.__parse_float(splitString[2])[0],
                                         self.__parse_float(splitString[3])[0],
                                         self.parse_string_input(" >|< ".join([splitString[4], splitString[5]]))[0],
                                         int(self.__parse_float(splitString[6])[0]))

        self.memory.populate_buffer(transitionData)
        return str(len(self.memory.buffer))

    def run(self):
        while True:
            time.sleep(0.01)  # Do you need to sleep for stability?
            if len(self.memory.buffer) > self.weightUpdateInterval:
                self.train()

    def train(self):
        """
        Call this method to train the NN after a number of steps have finished
        :return:
        """
        lastTransition = self.memory.buffer[-1]

        with tf.GradientTape() as tape:
            # Get the gradients of the local model
            loss = self._compute_loss(lastTransition)

        localGradients = tape.gradient(loss, self.localModel.trainable_weights)
        # Apply the local gradients to the global model
        self.optimiser.apply_gradients(zip(localGradients, self.globalModel.trainable_weights))
        # Update the local model
        self.localModel.set_weights(self.globalModel.get_weights())

        self.memory.clear_buffer()

    def _compute_loss(self, lastTransition):
        # If this is the terminal state
        if lastTransition.terminalState == 1:
            rewardSum = 0
        else:
            networkOutput = self.localModel.predict(lastTransition.newState)
            rewardSum = networkOutput[1]

        discountedRewards = []
        rewards = reversed([transition.reward for transition in self.memory.buffer])
        for reward in rewards:
            rewardSum = reward + (self.discountFactor * rewardSum)
            discountedRewards.append(rewardSum)

        discountedRewards = reversed(discountedRewards)

        # Compute the nn output over the whole batch/episode
        networkOutput = tf.convert_to_tensor(
            self.localModel.predict(np.array([transition.initialState for transition in self.memory.buffer])))

        # Calculate the value loss
        advantage = tf.convert_to_tensor(np.array(discountedRewards), dtype=tf.float32) - networkOutput[1]
        valueLoss = advantage ** 2

        # Calculate the policy loss
        oheAction = tf.one_hot([transition.action for transition in self.memory.buffer])

        # Adding entropy to the loss function discourages premature convergence
        entropy = tf.reduce_sum((networkOutput[0] * tf.math.log(networkOutput[0]) + 1e-20), axis=1)

        policyLoss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=oheAction, logits=networkOutput[0])
        policyLoss = policyLoss * tf.stop_gradient(advantage)
        policyLoss = policyLoss - 0.01 * entropy

        totalLoss = tf.reduce_mean((0.5 * valueLoss) + policyLoss)
        return totalLoss
