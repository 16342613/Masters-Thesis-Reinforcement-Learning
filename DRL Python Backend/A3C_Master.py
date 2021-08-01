from A3C_NN import A3C_NN
from A3C_Worker import A3C_Worker

import tensorflow as tf

class A3C_Master:
    def __init__(self, inputSize, actionCount):
        self.inputSize = inputSize
        self.actionCount = actionCount

        self.optimiser = tf.compat.v1.train.AdamOptimizer(0.0001, use_locking=True)
        self.globalModel = A3C_NN(self.inputSize, self.actionCount)
        self.workers = dict()


    def assign_worker(self, clientIP):
        worker = A3C_Worker(self.inputSize, self.actionCount, self.globalModel, clientIP, 0.99, self.optimiser)
        worker.run()

        self.workers[clientIP] = worker
        return worker


