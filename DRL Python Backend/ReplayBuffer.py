class ReplayBuffer:
    def __init__(self, filePath):
        self.filePath = filePath
        pass

    def read_buffer(self):
        with open(self.filePath, "r") as f:
            lines = f.readlines()

        return lines

    def sample_buffer(self):
        pass
