# -*- coding: utf-8 -*-
"""
Simple domains example
"""

import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.relpath("..//..//..//src//python"))
import hiped as hp

#%% 1) 0D domain

domain0 = hp.Domain(1) # a single dot
plt.clf()
domain0.plot()
plt.show()

#%% 2) 1D domain

domain1 = hp.Domain(2) # a 1D line segment has 2 vertices
plt.figure()
domain1.plot()
plt.show()

#%% 3) 2D domain

domain2 = hp.Domain(5) # a regular pentagon
plt.figure()
domain2.plot()
plt.show()

#%% 4) 3D domain

domain3 = hp.Domain("diamond6") # available : "tetra", "cube", "diamondN", "prismN" (N being an integer)
plt.figure()
domain3.plot()
plt.show()