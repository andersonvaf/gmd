import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_dict_equal

from .context import gmd


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_gmd_estimator(data):
    est = gmd.GMD(runs=1000, random_state=1234)
    assert est.alpha == 0.1
    assert est.runs == 1000

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')

    assert_dict_equal(est.subspaces_,  {0: [0, 2, 3], 1: [1, 3, 2], 2: [2, 3], 3: [3, 2]})
