import random


class ReplayBuffer:
    def __init__(self, savePath, capacity=100000):
        self.savePath = savePath
        self.capacity = capacity

        # The replay buffer is a FIFO queue. The queue goes from right to left (<--)
        self.buffer = []

        self.positiveBuffer = []
        self.negativeBuffer = []

    def populate_buffer(self, stateTransition):
        if len(self.buffer) == self.capacity:
            # Remove the last item in the queue
            self.buffer.pop(0)

        # Add the new item to the queue
        self.buffer.append(stateTransition)

        if len(self.positiveBuffer) > self.capacity:
            self.positiveBuffer.pop(0)
        if len(self.negativeBuffer) > self.capacity:
            self.negativeBuffer.pop(0)

        if stateTransition.reward > 0:
            self.positiveBuffer.append(stateTransition)
        else:
            self.negativeBuffer.append(stateTransition)

        # print(stateTransition.reward)
        # print("Positive samples: " + str(len(self.positiveBuffer)) + " ; Negative samples: " + str(len(self.negativeBuffer)) + " ; Total: " + str(len(self.buffer)))



    def sample_buffer(self, sampleSize):
        return random.sample(self.buffer, sampleSize)


    def prioritized_experience_sample(self, sampleSize, sampleSplit):
        positiveBufferSample = []
        negativeBufferSample = []

        try:
            positiveBufferSample.extend(random.sample(self.positiveBuffer, round(sampleSize * sampleSplit[0])))
        except ValueError:
            # If not enough positive transitions exist for sampling, randomly sample from both positive and negative transitions
            positiveBufferSample.extend(random.sample(self.positiveBuffer, len(self.positiveBuffer)))
            positiveBufferSample.extend(random.sample(self.buffer, round(sampleSize * sampleSplit[0]) - len(positiveBufferSample)))

        try:
            negativeBufferSample.extend(random.sample(self.negativeBuffer, round(sampleSize * sampleSplit[1])))
        except ValueError:
            # If not enough negative transitions exist for sampling, randomly sample from both positive and negative transitions
            negativeBufferSample.extend(random.sample(self.negativeBuffer, len(self.negativeBuffer)))
            negativeBufferSample.extend(negativeBufferSample.extend(random.sample(self.buffer, round(sampleSize * sampleSplit[1]) - len(negativeBufferSample))))

        positiveBufferSample.extend(negativeBufferSample)

        return positiveBufferSample


    def update_reward(self, stateID, newReward):
        matchedTransition = None

        for i in range(len(self.buffer)):
            if self.buffer[i].ID == stateID:
                self.buffer.pop(i)
                break

        for i in range(len(self.positiveBuffer)):
            if self.positiveBuffer[i].ID == stateID:
                matchedTransition = self.positiveBuffer.pop(i)
                break

        for i in range(len(self.negativeBuffer)):
            if self.negativeBuffer[i].ID == stateID:
                matchedTransition = self.negativeBuffer.pop(i)
                break

        matchedTransition.reward = newReward
        self.populate_buffer(matchedTransition)


