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

    def update_reward(self, stateID, newReward):
        # Get the index of the transition with the initial state which matches the state ID
        for i in range(len(self.buffer)):
            if self.buffer[i].ID == stateID:
                self.buffer[i].reward = newReward
                self.rewards[i] = newReward
                return
        # If a state transition with a matching ID could not be found (probably if the penetrating shot happened towards
        # the end of the maximum memory before the memory was cleared)
        # print("WARNING: Could not find state ID " + str(stateID))



    def clear_buffer(self):
        self.buffer = []
        self.initialStates = []
        self.actions = []
        self.rewards = []
        self.newStates = []

