import numpy as np


class NpQueue:
    """Queue for numpy arrays."""

    def __init__(self, max_len, elem_dim):
        self.q = np.zeros((max_len, elem_dim))

        self.currSize = 0
        self.maxQLen = max_len
        self.elemDim = elem_dim

    def push(self, x):
        """Adding element to queue."""
        self.q[1:] = self.q[:-1]
        self.q[0] = x

        if self.currSize < self.maxQLen:
            self.currSize += 1

    def __len__(self):
        return len(self.q)
