from A3C_NN import A3C_NN
from A3C_Buffer import A3C_Buffer
from StateTransition import StateTransition
from HelperPy import HelperPy

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

        self.helper = HelperPy()

        self.localModel = A3C_NN(self.inputSize, self.actionCount)
        self.memory = A3C_Buffer()

        self.weightUpdateInterval = weightUpdateInterval
        self.acceptAppends = False


    def append_to_buffer(self, stringInput):
        if self.acceptAppends is True:
            transitionData = StateTransition.string_to_transition(stringInput)
            self.memory.populate_buffer(transitionData)

            return str(len(self.memory.buffer))
        else:
            return "0"

    def run(self):
        print("Started worker thread")
        while True:
            # print(len(self.memory.buffer))
            time.sleep(0.001)  # Do you need to sleep for stability? --> YES
            if len(self.memory.buffer) >= self.weightUpdateInterval:
                self.acceptAppends = False
                self.train()
            else:
                self.acceptAppends = True

    def train(self):
        """
        Call this method to train the NN after a number of steps have finished
        :return:
        """
        lastTransition = self.memory.buffer[-1]

        with tf.GradientTape(persistent=True) as tape:
            # Get the gradients of the local model
            loss = self._compute_loss(lastTransition, self.memory, self.discountFactor)

        # print(loss)
        localGradients = tape.gradient(loss, self.localModel.trainable_weights)
        # Apply the local gradients to the global model
        self.optimiser.apply_gradients(zip(localGradients, self.globalModel.trainable_weights))
        # Update the local model
        self.localModel.set_weights(self.globalModel.get_weights())
        self.memory.clear_buffer()


    def _compute_loss(self, lastTransition, memory, discountFactor):
        # If this is the terminal state
        if lastTransition.terminalState == 1:
            rewardSum = 0
        else:
            # networkOutput = self.localModel.get_prediction(tf.convert_to_tensor(np.array([lastTransition.newState])))
            networkOutput = self.localModel(tf.convert_to_tensor([lastTransition.newState], dtype=tf.float32))
            rewardSum = networkOutput[1].numpy()[0][0]


        discountedRewards = []
        # rewards = [transition.reward for transition in memory.buffer][::-1]
        for reward in memory.rewards[::-1]:
            rewardSum = reward + (discountFactor * rewardSum)
            discountedRewards.append(rewardSum)

        discountedRewards.reverse()

        # Compute the nn output over the whole batch/episode
        networkOutput = self.localModel(tf.convert_to_tensor(memory.initialStates, dtype=tf.float32))

        # Calculate the value loss
        advantage = tf.convert_to_tensor(discountedRewards, dtype=tf.float32) - networkOutput[1]
        valueLoss = advantage ** 2

        # Calculate the policy loss
        oheAction = tf.one_hot(memory.actions, self.actionCount, dtype=tf.float32)

        # Adding entropy to the loss function discourages premature convergence
        policy = networkOutput[0]
        entropy = tf.reduce_sum(networkOutput[0] * tf.math.log(networkOutput[0] + 1e-20), axis=1)

        policyLoss = tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2(labels=oheAction, logits=networkOutput[0])
        policyLoss = policyLoss * tf.stop_gradient(advantage)
        policyLoss = policyLoss - 0.01 * entropy

        totalLoss = tf.reduce_mean((0.5 * valueLoss) + policyLoss)
        return totalLoss


