.. StreamGMD documentation master file, created by
   sphinx-quickstart on Mon May 20 20:04:41 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to StreamGMD's documentation!
=====================================

This project provides a scikit-learn compatible python implementation of the
algorithm presented in [`Trittenbach2018`_] together with some usage examples
and a reproduction of the results from the paper.

Recent approaches in outlier detection seperate the subspace search from the
actual outlier detection and run the outlier detection algorithm on a
projection of the original feature space. See [`Keller2012`_]. As a result the
detection algorithm (Local Outlier Factor is used in the paper) does not suffer
from the curse of dimensionality.


.. _Trittenbach2018: https://link.springer.com/article/10.1007/s41060-018-0137-7
.. _Keller2012: https://ieeexplore.ieee.org/document/6228154

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   quick_start
   user_guide

