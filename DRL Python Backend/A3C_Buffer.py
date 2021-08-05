from StateTransition import StateTransition

class A3C_Buffer:
    def __init__(self):
        self.buffer = []
        self.initialStates = []
        self.actions = []
        self.rewards = []
        self.newStates = []


    def populate_buffer(self, stateTransition):
        self.buffer.append(stateTransition)

        self.initialStates.append(stateTransition.initialState)
        self.actions.append(int(stateTransition.action))
        self.rewards.append(stateTransition.reward)
        self.newStates.append(stateTransition.newState)

        # if stateTransition.reward > 0:
        #     self.positiveBuffer.append(stateTransition)
        # else:
        #     self.negativeBuffer.append(stateTransition)


    def clear_buffer(self):
        self.buffer = []
        self.initialStates = []
        self.actions = []
        self.rewards = []
        self.newStates = []

