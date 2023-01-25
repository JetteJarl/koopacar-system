class FlexibleQueue:
    """ Queue storing arbitrary elements. """

    def __init__(self, max_length):
        self.queue = []

        self.size = 0
        self.max_length = max_length

    def push(self, x):
        """ Inserts a new element x into the list. """
        self.queue.insert(0, x)

        self.size += 1

        if self.size > self.max_length:
            self.queue.pop(self.max_length)
            self.size -= 1

    def get(self, index):
        """ Returns element at index 'index' """
        return self.queue[index]

    def __len__(self):
        return len(self.queue)
