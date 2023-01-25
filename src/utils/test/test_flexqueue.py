import unittest
from src.utils.flex_queue import FlexibleQueue


class FlexibleQueueTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.MAX_Q_LEN = 5
        self.test_queue = FlexibleQueue(self.MAX_Q_LEN)

    def tearDown(self):
        pass

    def test_push(self):
        # Test adding single element
        test_element = [0, 0.0, 0.0]
        self.test_queue.push(test_element)

        self.assertEqual([test_element], self.test_queue.queue)

        # Test adding multiple elements leading to overflow
        test_overflow_list = [[1, 1.1, 1.1], [2, 2.2, 2.2], [3, 3.3, 3.3], [4, 4.4, 4.4], [5, 5.5, 5.5]]
        expected_content = test_overflow_list.copy()
        expected_content.reverse()

        for element in test_overflow_list:
            self.test_queue.push(element)

        self.assertEqual(expected_content, self.test_queue.queue)

    def test_get(self):
        self.assertRaises(IndexError, self.test_queue.get, self.MAX_Q_LEN)


if __name__ == '__main__':
    unittest.main()
