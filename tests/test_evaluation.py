import unittest
import numpy as np
import pandas as pd

from .context import *


class TestEvaluation(unittest.TestCase):
    def test_compute_jaccard_coeff(self):
        s1, s2, s3 = set([0, 1]), set([0, 2]), set([2, 3])

        np.testing.assert_almost_equal(jaccard_coeff(s1, s2), 1 / 3)
        np.testing.assert_almost_equal(jaccard_coeff(s1, s1), 1.0)
        np.testing.assert_almost_equal(jaccard_coeff(s1, s3), 0.0)

    def test_compute_sim_matrix(self):
        s1, s2, s3 = set([0, 1]), set([0, 2]), set([2, 3])
        df1 = pd.DataFrame({"C0": [s1, s2], "C1": [s2, s3], "C2": [s3, s1]})
        df2 = pd.DataFrame({"C0": [s1, s2], "C1": [s3, s1], "C2": [s1, s2]})

        np.testing.assert_array_almost_equal(
            similarity_matrix(df1, df2),
            np.array([[1.0, 1 / 3, 0.0], [1.0, 0.0, 1 / 3]]),
        )

