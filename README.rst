.. -*- mode: rst -*-

|Travis|_ |AppVeyor|_ |Codecov|_ |CircleCI|_ |ReadTheDocs|_

.. |Travis| image:: https://travis-ci.org/scikit-learn-contrib/project-template.svg?branch=master
.. _Travis: https://travis-ci.org/scikit-learn-contrib/project-template

.. |AppVeyor| image:: https://ci.appveyor.com/api/projects/status/coy2qqaqr1rnnt5y/branch/master?svg=true
.. _AppVeyor: https://ci.appveyor.com/project/glemaitre/project-template

.. |Codecov| image:: https://codecov.io/gh/scikit-learn-contrib/project-template/branch/master/graph/badge.svg
.. _Codecov: https://codecov.io/gh/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/scikit-learn-contrib/project-template.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/scikit-learn-contrib/project-template/tree/master

.. |ReadTheDocs| image:: https://readthedocs.org/projects/sklearn-template/badge/?version=latest
.. _ReadTheDocs: https://sklearn-template.readthedocs.io/en/latest/?badge=latest

Scikit-learn Greedy Maximum Deviation (GMD) Algorithm
=====================================================

.. _scikit-learn: https://scikit-learn.org

This project provides a `scikit-learn`_ compatible python implementation of the
algorithm presented in [`Trittenbach2018`_] together with some usage examples
and a reproduction of the results from the paper.

Recent approaches in outlier detection seperate the subspace search from the
actual outlier detection and run the outlier detection algorithm on a
projection of the original feature space. See [`Keller2012`_]. As a result the
detection algorithm (Local Outlier Factor is used in the paper) does not suffer
from the curse of dimensionality.


.. _Trittenbach2018: https://link.springer.com/article/10.1007/s41060-018-0137-7
.. _Keller2012: https://ieeexplore.ieee.org/document/6228154

.. _documentation: https://sklearn-template.readthedocs.io/en/latest/quick_start.html

Refer to the documentation_ to see usage examples.

