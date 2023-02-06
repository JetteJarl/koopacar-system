import unittest
import numpy as np
import numpy.testing
from src.utils.np_queue import NpQueue


class NpQueueTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.MAX_Q_LEN = 5
        self.ELEM_DIM = 2
        self.tested_queue = NpQueue(self.MAX_Q_LEN, self.ELEM_DIM)

    def tearDown(self):
        pass

    def test_addElement(self):
        # create input
        input_elem = [1, 1]
        # add input to queue
        self.tested_queue.push(input_elem)
        # test queue
        np.testing.assert_allclose(input_elem, self.tested_queue.q[0])

    def test_addElementOverflow(self):
        # create input/output
        input_elem = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]]
        overflow_elem = [5, 5]
        output_elem = [[5, 5], [4, 4], [3, 3], [2, 2], [1, 1]]
        # add input to queue
        [self.tested_queue.push(elem) for elem in input_elem]
        self.tested_queue.push(overflow_elem)
        # test queue
        self.assertEqual(self.MAX_Q_LEN, len(self.tested_queue.q))
        np.testing.assert_allclose(output_elem, self.tested_queue.q)


if __name__ == '__main__':
    unittest.main()
