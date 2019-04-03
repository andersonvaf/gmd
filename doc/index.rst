Greedy Maximum Deviation
========================

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
   :maxdepth: 1
   :hidden:
   :caption: Getting Started

   quick_start

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Documentation

   api
   user_guide

`Getting started <quick_start.html>`_
-------------------------------------

Information regarding this template and how to modify it for your own project.

`User Guide <user_guide.html>`_
-------------------------------

An example of narrative documentation.

`API Documentation <api.html>`_
-------------------------------

An example of API documentation.