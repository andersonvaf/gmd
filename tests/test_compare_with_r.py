import os
import unittest
import pandas as pd
import numpy as np
import numpy.testing as npt

from .context import *
from .context import libgmdc


class TestGMD(unittest.TestCase):
    """
    Test the GMD implementation. The implementation is based on the original
    implementation from the paper: https://github.com/holtri/R-subcon
    """

    def test_create_sorted(self):
        unsorted = np.array([[0, 3, 2], [1, 0, 3], [3, 2, 0]])
        greedy = GMD(random_state=1234)
        greedy.fit(unsorted)
        self.assertEqual(greedy._sorted.tolist(), [[0, 1, 2], [1, 2, 0], [2, 0, 1]])

    def test_kstest_with_ties(self):
        """
        Compare the output of the cython kstest implementation to the output of
        the original implementation.
        """
        view = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
            ],
            dtype=np.uint8,
        )
        data = np.array(
            [
                0.00,
                0.21,
                0.06,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
                0.15,
                0.06,
                0.00,
                0.00,
                0.00,
                0.00,
                0.00,
            ]
        )
        sorted_index = np.argsort(data, kind="mergesort")
        distance = libgmdc.kstest(view, sorted_index.astype(np.int32))
        self.assertAlmostEqual(distance, 0.466666666666667)

    def test_kstest_without_ties(self):
        """
        Compare the output of the cython kstest implementation to the output of
        the original implementation.
        """
        view = np.array(
            [
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
                True,
                False,
            ],
            dtype=np.uint8,
        )
        data = np.array(
            [
                0.01,
                0.21,
                0.06,
                0.02,
                0.03,
                0.04,
                0.05,
                0.07,
                0.15,
                0.08,
                0.09,
                0.10,
                0.11,
                0.12,
                0.13,
            ]
        )
        sorted_index = np.argsort(data, kind="mergesort")
        distance = libgmdc.kstest(view, sorted_index.astype(np.int32))
        self.assertAlmostEqual(distance, 0.6)

    def test_compare_kstest_with_r(self):
        dist = np.array([1, 1, 4, 1, 9])
        selection = np.array([False, True, False, False, False], dtype=np.uint8)
        sort = np.argsort(dist)
        self.assertEqual(libgmdc.kstest(selection, sort.astype(np.int32)), 0.6)

    def test_compare_kstest_with_r_two_selected(self):
        dist = np.array([1, 1, 4, 1, 9])
        selection = np.array([False, True, True, False, False], dtype=np.uint8)
        sort = np.argsort(dist)
        self.assertEqual(libgmdc.kstest(selection, sort.astype(np.int32)), 0.2)

    def test_compare_kstest_with_r_all_selected(self):
        dist = np.array([1, 1, 4, 1, 9])
        selection = np.array([True, True, True, True, True], dtype=np.uint8)
        sort = np.argsort(dist)
        self.assertAlmostEqual(libgmdc.kstest(selection, sort.astype(np.int32)), 0.0)

    @unittest.skipIf(
        "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
        "Skipping this test on Travis CI.",
    )
    def test_compare_kstest_with_slice_from_r(self):
        r_slice = pd.read_csv("tests/res/slice_from_r_0.71789.csv", header=None).values
        r_slice = r_slice[:, 0].astype(np.uint8)

        data = pd.read_csv(
            "tests/res/spambase_small.data", index_col=None, header=None
        ).values
        greedy = GMD(random_state=1234).fit(data)

        self.assertAlmostEqual(libgmdc.kstest(r_slice, greedy._sorted[:, 0]), 0.7178874)

    @unittest.skipIf(
        "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
        "Skipping this test on Travis CI.",
    )
    def test_sorted_is_the_same_as_in_r(self):
        compare = np.array(
            [
                [0, 3, 3, 3, 10],
                [3, 4, 4, 4, 17],
                [4, 5, 5, 5, 20],
                [5, 6, 6, 6, 24],
                [6, 7, 7, 7, 26],
                [7, 10, 10, 10, 27],
                [10, 11, 13, 13, 28],
                [11, 13, 16, 16, 33],
                [12, 14, 17, 17, 55],
                [13, 16, 20, 20, 56],
            ]
        )
        data = pd.read_csv(
            "tests/res/spambase_small.data", index_col=None, header=None
        ).values
        greedy = GMD(random_state=1234).fit(data)
        npt.assert_array_equal(greedy._sorted[0:10], compare)

    @unittest.skipIf(
        "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
        "Skipping this test on Travis CI.",
    )
    def test_avg_deviation_statistics(self):
        """
        Compares the output from the R deviation computation to the output of
        the new implementation by using the data in res/dt_uniform.csv.

        The csv is created using R and the following commands:

        library(data.table)
        library(subcon)
        dt <- fread('tests/res/spambase_small.data')
        indexMatrix <- sortedIndexMatrix(dt)
        out <- deviationStatisticsC(indexMap = indexMatrix, alpha=0.1, numRuns=10000)['avg']
        write.csv(out, file='tests/res/deviations_compare_with_R.csv')
        """
        comp = pd.read_csv(
            "tests/res/deviations_compare_with_R.csv", index_col=0
        ).values
        data = pd.read_csv(
            "tests/res/spambase_small.data", index_col=None, header=None
        ).values
        greedy = GMD(runs=1000, random_state=1234).fit(data)
        res = greedy._deviation_matrix()

        npt.assert_almost_equal(comp, res, decimal=2)

    @unittest.skipIf(
        "TRAVIS" in os.environ and os.environ["TRAVIS"] == "true",
        "Skipping this test on Travis CI.",
    )
    def test_gmd_1(self):
        data = pd.read_csv(
            "tests/res/spambase_small.data", index_col=None, header=None
        ).values

        greedy = GMD(alpha=0.1, runs=1000, random_state=1234)
        greedy.fit(data)
        subspaces, _ = greedy._max_deviation_subspaces(4)
        self.assertEqual(subspaces, [4, 3, 2])  # computed with R impl


if __name__ == "__main__":
    unittest.main()
