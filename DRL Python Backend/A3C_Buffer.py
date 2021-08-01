from StateTransition import StateTransition

class A3C_Buffer:
    def __init__(self):
        self.buffer = []
        self.positiveBuffer = []
        self.negativeBuffer = []


    def populate_buffer(self, stateTransition):
        self.buffer.append(stateTransition)

        if stateTransition.reward > 0:
            self.positiveBuffer.append(stateTransition)
        else:
            self.negativeBuffer.append(stateTransition)


    def clear_buffer(self):
        self.buffer = []
        self.positiveBuffer = []
        self.negativeBuffer = []

