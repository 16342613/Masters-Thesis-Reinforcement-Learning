from queue import Queue

from A3C_NN import A3C_NN
from A3C_Worker import A3C_Worker
from HelperPy import HelperPy

import tensorflow as tf
import keras
import os
import pickle
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from datetime import datetime
from Global import Global
import numpy as np

#from Test_A3C_MOD import Worker
# from Test_A3C_MOD import ActorCriticModel


class A3C_Master:
    def __init__(self, inputSize, actionCount):
        self.inputSize = inputSize
        self.actionCount = actionCount

        self.optimiser = tf.compat.v1.train.AdamOptimizer(0.0001, use_locking=True)
        self.pretrainedWeightsAvailable = False
        self.setWeights = False
        self.globalSavePath = "Generated Data/Saved Models/"
        self.globalModel = None

        if os.path.isfile(os.path.join(self.globalSavePath, "test.pkl")):
            self.globalModel = A3C_NN(self.inputSize, self.actionCount)
            self.globalModel.build((None, self.inputSize))

            with open(self.globalSavePath + "test.pkl", "rb") as file:
                weights = pickle.load(file)
                for i in range(len(weights)):
                    self.globalModel.trainable_variables[i].assign(weights[i])
            print("Loaded saved weights")

        else:
            # self.globalModel = A3C_NN(self.inputSize, self.actionCount)
            self.globalModel = A3C_NN(self.inputSize, self.actionCount)  # global network
            self.setWeights = True

        self.globalModel(tf.convert_to_tensor(np.random.random((1, self.inputSize)), dtype=tf.float32))

        self.workers = dict()
        self.currentRewards = []
        self.receivedPlots = []

    def assign_worker(self, clientIP):
        # worker = A3C_Worker(self.inputSize, self.actionCount, clientIP, 0.99, self.optimiser, weightUpdateInterval=15)
        worker = A3C_Worker(self.inputSize, self.actionCount, self.globalModel, self.optimiser, Queue(), "yo",
                        game_name="CartPole-v0", save_dir="Generated Data/Saved Models")

        # worker.start()

        self.currentRewards.append([])
        self.receivedPlots.append(False)

        # self.workers[clientIP] = worker
        return worker

    def global_predict(self, stringInput, parseString=True):
        helper = HelperPy()
        parsedInput = helper.parse_string_input(stringInput)
        outputs = Global.globalModel.get_prediction(parsedInput, parseString)

        return outputs

    def save_network(self, empty):
        # Global.globalModel.save_weights(os.path.join(self.globalSavePath, "globalModel.h5"))
        with open(self.globalSavePath + "test.pkl", "wb") as file:
            pickle.dump(self.globalModel.trainable_variables, file)
        print(datetime.now().strftime("%H:%M:%S") + " : Saved global model")

        return "-1"

    def plot_progress(self, stringInput):
        workerIndex = int(stringInput.split(" >|< ")[0])
        splitString = stringInput.split(" >|< ")[1].split(" | ")

        for i in range(len(splitString)):
            self.currentRewards[workerIndex].append(float(splitString[i]))

        self.receivedPlots[workerIndex] = True
        print("Received plots for worker " + str(workerIndex))

        # ySmooth = savgol_filter(self.currentRewards, 3, 3)
        # axes = plt.gca()
        # axes.set_ylim([0, 50])

        x = [i for i in range(len(self.currentRewards[workerIndex]))]
        plt.plot(x, self.currentRewards[workerIndex], label=str(workerIndex))

        if os.path.isfile("Generated Data/Screenshots/latestPlot.png"):
            os.remove("Generated Data/Screenshots/latestPlot.png")
        plt.savefig("Generated Data/Screenshots/latestPlot.png")
        plt.clf()

        # if all(self.receivedPlots) is True:
        #     for i in range(len(self.currentRewards)):
        #         x = [i for i in range(len(self.currentRewards[workerIndex]))]
        #         plt.plot(x, self.currentRewards[workerIndex], label=str(i))
        #     self.receivedPlots = [False for i in range(len(self.receivedPlots))]
        #
        #     if os.path.isfile("Generated Data/Screenshots/latestPlot.png"):
        #         os.remove("Generated Data/Screenshots/latestPlot.png")
        #     plt.savefig("Generated Data/Screenshots/latestPlot.png")
        #     plt.clf()
        # #plt.plot(x, ySmooth, color='black')
        print(datetime.now().strftime("%H:%M:%S") + " : Saved plots")

        return "1.0"
