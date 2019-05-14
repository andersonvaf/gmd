import unittest
import numpy as np

from gmd import IncSortedIndex

class TestIncGMD(unittest.TestCase):
    """
    Test the incremental GMD implementation
    """

    def setUp(self):
        unsorted = np.array([[0.1, 1.4, 2.7],                             
                             [1.2, 1.5, 0.8],                             
                             [2.3, 3.6, 1.9]])
        self.sorted_index = IncSortedIndex(unsorted)

    def test_can_create_sorted(self):
        self.assertListEqual(self.sorted_index.sorted.tolist(), [[0, 0, 1], [1, 1, 2], [2, 2, 0]])

    def test_delete_two_oldest(self):
        e1 = np.array([4.8, 0.2, 1.2])
        e2 = np.array([1.5, 2.4, 0.3])
        self.sorted_index.del_and_ins_sorted(e1)
        self.sorted_index.del_and_ins_sorted(e2)
        self.assertListEqual(self.sorted_index.sorted.tolist(), [[2, 1, 2], [0, 2, 1], [1, 0, 0]])

    def test_delete_and_insert(self):
        e1 = np.array([4.8, 0.2, 1.2])
        self.sorted_index.del_and_ins_sorted(e1)
        self.assertListEqual(self.sorted_index.sorted.tolist(), [[0, 2, 0], [1, 0, 2], [2, 1, 1]])

    def test_delet_and_insert_multiple_times(self):
        e1 = np.array([4.8, 0.2, 1.2])
        e2 = np.array([1.5, 2.4, 0.3])
        self.sorted_index.del_and_ins_sorted(e1)
        self.sorted_index.del_and_ins_sorted(e2)
        self.assertListEqual(self.sorted_index.sorted.tolist(), [[2, 1, 2], [0, 2, 1], [1, 0, 0]])

if __name__ == '__main__':
    unittest.main()
