"""
===========================
Plotting Template Estimator
===========================

An example plot of :class:`gmd.template.TemplateEstimator`
"""
import numpy as np
from matplotlib import pyplot as plt
from gmd import GMD

X = np.arange(100).reshape(100, 1)
y = np.zeros((100, ))
estimator = GMD()
estimator.fit(X, y)
#plt.plot(estimator.predict(X))
#plt.show()
