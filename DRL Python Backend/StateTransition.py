class StateTransition:
    def __init__(self, initialState, action, reward, newState, ID):
        self.initialState = initialState
        self.action = action
        self.reward = reward
        self.newState = newState
        self.ID = ID
