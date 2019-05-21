import pytest

from sklearn.utils.estimator_checks import check_estimator

from .context import *


@pytest.mark.parametrize("Estimator", [GMD])
def test_all_estimators(Estimator):
    return check_estimator(Estimator)
