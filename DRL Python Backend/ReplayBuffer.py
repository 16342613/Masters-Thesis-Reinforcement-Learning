import random

class ReplayBuffer:
    def __init__(self, savePath, capacity = 500):
        self.savePath = savePath
        self.capacity = capacity

        # The replay buffer is a FIFO queue. The queue goes from right to left (<--)
        self.buffer = []


    def populate_buffer(self, stateTransition):
        if len(self.buffer) == self.capacity:
            # Remove the last item in the queue
            self.buffer.pop(0)
        else:
            # Add the new item to the queue
            self.buffer.append(stateTransition)



    def sample_buffer(self, sampleSize):
        return random.sample(self.buffer, sampleSize)
