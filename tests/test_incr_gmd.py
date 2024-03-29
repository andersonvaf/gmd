import unittest
import numpy as np
import pandas as pd

from .context import *
from .context import libgmdc


class TestIncGMD(unittest.TestCase):
    """
    Test the incremental GMD implementation
    """

    def gen_data(self, length):
        return pd.DataFrame(
            {
                "x1": np.random.uniform(0, 1, length),
                "x2": np.random.uniform(0, 1, length),
                "x3": np.random.uniform(0, 1, length),
                "x4": np.random.uniform(0, 1, length),
                "x5": np.random.uniform(0, 1, length),
                "x6": np.random.uniform(0, 1, length),
                "x7": np.random.uniform(0, 1, length),
                "x8": np.random.uniform(0, 1, length),
                "x9": np.random.uniform(0, 1, length),
                "x10": np.random.uniform(0, 1, length),
            }
        )

    def setUp(self):
        unsorted = np.array([[0.1, 1.4, 2.7], [1.2, 1.5, 0.8], [2.3, 3.6, 1.9]])

        self.sorted_index = IncSortedIndex(unsorted)

    def test_can_create_sorted(self):
        self.assertListEqual(
            self.sorted_index.sorted.tolist(), [[0, 0, 1], [1, 1, 2], [2, 2, 0]]
        )

    def test_delete_two_oldest(self):
        e1 = np.array([4.8, 0.2, 1.2])
        e2 = np.array([1.5, 2.4, 0.3])
        self.sorted_index.del_and_ins_sorted(e1)
        self.sorted_index.del_and_ins_sorted(e2)
        self.assertListEqual(
            self.sorted_index.sorted.tolist(), [[2, 1, 2], [0, 2, 1], [1, 0, 0]]
        )

    def test_delete_and_insert(self):
        e1 = np.array([4.8, 0.2, 1.2])
        self.sorted_index.del_and_ins_sorted(e1)
        self.assertListEqual(
            self.sorted_index.sorted.tolist(), [[0, 2, 0], [1, 0, 2], [2, 1, 1]]
        )

    def test_delete_and_insert_multiple_times(self):
        e1 = np.array([4.8, 0.2, 1.2])
        e2 = np.array([1.5, 2.4, 0.3])
        self.sorted_index.del_and_ins_sorted(e1)
        self.sorted_index.del_and_ins_sorted(e2)
        self.assertListEqual(
            self.sorted_index.sorted.tolist(), [[2, 1, 2], [0, 2, 1], [1, 0, 0]]
        )

    def test_inc_insert_is_equal_to_regular_sorting(self):
        df = self.gen_data(1000)
        window_size = 50
        start_window = df[:window_size]

        sorted_index = IncSortedIndex(start_window)
        for i in range(window_size, len(df)):
            sorted_index.del_and_ins_sorted(df.iloc[i])

        # the sorted_index should now equal the data structure we obtain by just
        # sorting the last 100 elements

        reference = GMD().create_sorted_index(df[-window_size:])
        self.assertListEqual(sorted_index.sorted.tolist(), reference.tolist())

    def test_initial_sorting_works_with_ties(self):
        data = np.array(
            [
                [0.0, 0.64, 0.64, 0.0, 0.32],
                [0.21, 0.28, 0.5, 0.0, 0.14],
                [0.06, 0.0, 0.71, 0.0, 1.23],
                [0.0, 0.0, 0.0, 0.0, 0.63],
                [0.0, 0.0, 0.0, 0.0, 0.63],
            ]
        )
        expected = GMD().create_sorted_index(data)
        actual = IncSortedIndex(data).sorted
        self.assertListEqual(actual.tolist(), expected.tolist())


if __name__ == "__main__":
    unittest.main()
